"""Joint human-data extension of the unchanged MIT trainer.

Concatenate the existing MIT training dataset with ScanDiff's COCO train
scanpaths, shuffled uniformly over records, and keep MIT validation unchanged.
The two narrow hooks below add loader/config metadata without modifying the
core source files frozen for the MIT-only test evaluation.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import src.data.mit1003_dataset as mit

COCO = Path('/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/data')
SOURCE = COCO / 'COCOFreeView_fixations_train.json'


class COCOTrain(Dataset):
    def __init__(self, max_seq_len=16):
        self.max_seq_len=max_seq_len
        self.records=json.loads(SOURCE.read_text())
        assert len(self.records)==30129
        assert all(r['split']=='train' for r in self.records)
        self.names=sorted({r['name'] for r in self.records})
        assert len(self.names)==3021
        self.ids={name:1003+i for i,name in enumerate(self.names)}
        for name in self.names:
            path=COCO/'images'/name
            with Image.open(path) as img:
                assert img.size==(1680,1050), (path,img.size)
        for r in self.records:
            assert 0<len(r['X'])==len(r['Y'])==len(r['T'])
            assert np.isfinite(r['X']).all() and np.isfinite(r['Y']).all()
        # The COCO-FreeView json already contains the initial central
        # fixation (first fixation within 0.1 of the centre in 97.4 % of scanpaths). Our
        # loader prepends its own centre fixation (MIT convention: pysaliency data come
        # WITHOUT it), so the json's first fixation must be dropped, otherwise the first
        # target is "stay at the centre". Scanpaths with no fixation after the start
        # (33 of 30,129) yield no prediction step and are removed.
        self.records=[r for r in self.records if len(r['X'])>=2]
        assert len(self.records)==30129-33, len(self.records)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(COCO/'images'/r['name']) as img:
            image=mit._img_transform(img.convert('RGB'))
        X,Y=r['X'][1:],r['Y'][1:]          # drop the json's initial central fixation (see __init__)
        n=min(len(X),self.max_seq_len-1)
        f=torch.zeros(self.max_seq_len,2,dtype=torch.float32)
        # Original-pixel convention of our MIT loader, on verified 1680x1050 images.
        f[1:n+1,0]=torch.tensor(np.clip(np.asarray(X[:n])/1679*2-1,-1,1),dtype=torch.float32)
        f[1:n+1,1]=torch.tensor(np.clip(np.asarray(Y[:n])/1049*2-1,-1,1),dtype=torch.float32)
        return dict(image=image,fixations=f,fixations_len=torch.tensor(n+1),
                    img_idx=torch.tensor(self.ids[r['name']]))


def install_joint_loader():
    original=mit.make_mit1003_loader
    def loader(*args,**kwargs):
        base=original(*args,**kwargs)
        if not kwargs.get('shuffle',False):return base
        coco=COCOTrain(kwargs.get('max_seq_len',16))
        mit_names={Path(base.dataset._filenames[i]).name.lower()
                   for i in {r['img_idx'] for r in base.dataset._records}}
        assert not mit_names & {n.lower() for n in coco.names}
        joint=ConcatDataset([base.dataset,coco])
        print(f'JOINT TRAIN: MIT {len(base.dataset)} paths + COCO {len(coco)} paths = {len(joint)}; uniform record shuffle',flush=True)
        return DataLoader(joint,batch_size=kwargs['batch_size'],shuffle=True,
                          num_workers=kwargs['num_workers'],pin_memory=True,
                          generator=torch.Generator().manual_seed(kwargs['seed']))
    mit.make_mit1003_loader=loader


def main():
    assert os.environ.get('OURS_CROP_TRANSFORM','0')!='1'
    if '--check-only' in sys.argv:
        c=COCOTrain()
        sample=c[0]
        assert sample['image'].shape==(3,224,224)
        assert sample['fixations'].shape==(16,2)
        assert sample['fixations'][0].eq(0).all()
        assert sample['fixations'].abs().max()<=1
        assert sample['fixations_len'].item()==min(len(c.records[0]['X']),16)   # start fixation dropped, centre prepended
        print('COCO TRAIN PREFLIGHT PASSED:',len(c),'paths;',len(c.names),'images; geometry/finite/padding checks passed')
        return
    if not torch.cuda.is_available():raise RuntimeError('GPU required for training')
    import src.training.train as train
    original_parse=train.parse_args
    def parse():
        a=original_parse()
        assert a.dataset=='mit1003' and a.mit_split_dir and a.cv_fold is None
        assert a.resume is None
        a.human_training_data='MIT1003_train+COCOFreeView_train'
        a.coco_train_json=str(SOURCE)
        a.coco_train_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        a.joint_sampling='uniform_scanpath_concat'
        a.coco_geometry='1680x1050; normalize by W-1,H-1; clip; Resize224'
        a.coco_initial_fixation='prepend centre as model start token; retain first 15 recorded fixations'
        a.joint_validation='MIT1003 validation only'
        a.joint_entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return a
    train.parse_args=parse
    install_joint_loader()
    train.main()


if __name__=='__main__':main()
