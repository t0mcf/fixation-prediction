"""Freeze validation-only ScanDiff selections; gated test sampling via authors' launcher."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SD=Path('/mnt/vast-nhr/projects/nim00018/tom/scandiff')
SPLITS=Path('/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003')
COMPONENTS=('MM_Vector','MM_Direction','MM_Length','MM_Position')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def scores(p):
    d=json.loads(Path(p).read_text())
    vals=[float(d[k]['KLD']) for k in COMPONENTS]
    sm=float(d['ScanMatchNoDur']['KLD'])
    require(all(math.isfinite(x) and x>=0 for x in vals+[sm]),'Invalid duration-free metrics')
    return sum(vals)/4,sm

def choose(candidates):
    def rank(name,i):
        return sum(v['scores'][i]<candidates[name]['scores'][i] for v in candidates.values())
    def key(n):
        return ((rank(n,0)+rank(n,1))/2, int(n.split('_')[1]) if n.startswith('epoch_') else 10**6)
    return min(candidates,key=key)

def verify(m):
    for p,h in m['files'].items():require(sha(p)==h,'Frozen file changed: '+p)
    for r in m['records']:
        require(choose(r['candidates'])==r['selected'],'Selection changed')
        for c in r['candidates'].values():require(list(scores(c['metrics']))==c['scores'],'Validation scores changed')

def build(path):
    import yaml
    require(not path.exists(),'Manifest exists')
    files={}
    for folder,pattern in [(SD/'src','*.py'),(SD/'configs','*.yaml')]:
        for p in folder.rglob(pattern):files[str(p)]=sha(p)
    for p in [Path(__file__).resolve(),SD/'slurm/score_ft_valsplit.sh',ROOT/'slurm/frozen_scandiff_transfer.sbatch',SD/'data/task_embeddings.npy']:
        files[str(p)]=sha(p)
    records=[]
    names=[f'{arm}_seed{s}' for arm in ['ft_mit_scratch','ft_mit_av1k','ft_mit_av10k','ft_mit_av50k','ft_mit_av100k','r1_official_recipe','armB_joint_avinit'] for s in [0,1]]
    for run in names:
        candidates={}
        for d in sorted((SD/'data/eval/ft_valscore').glob(run+'_*_val')):
            label=d.name[len(run)+1:-4]
            require(re.fullmatch(r'epoch_\d+|last',label),'Unexpected checkpoint label')
            metrics=list(d.glob('results/metrics_epoch_*/metrics_MIT1003Dataset_test.json'))
            require(len(metrics)==1,'Missing/ambiguous validation metrics: '+str(d))
            cfgpath=d/'hydra/.hydra/config.yaml';cfg=yaml.safe_load(cfgpath.read_text())
            require(cfg['data']['test_datasets']['mit1003']['split']=='valid','Candidate is not validation')
            require(cfg['diffusion']['num_timesteps']==1000 and cfg['seed']==0,'Different sampling protocol')
            require(cfg['data']['batch_size_test']==1,'Different evaluation batch size')
            cp=Path(cfg['ckpt_path'])
            require(cp.name==label+'.pt' and cp.parents[2].name==run,'Checkpoint label/path mismatch')
            candidates[label]=dict(scores=list(scores(metrics[0])),metrics=str(metrics[0]),config=str(cfgpath),checkpoint=str(cp))
            for p in [metrics[0],cfgpath]:files[str(p)]=sha(p)
        require(candidates,'No candidates: '+run)
        winner=choose(candidates);cp=Path(candidates[winner]['checkpoint'])
        files[str(cp)]=sha(cp)
        records.append(dict(id=run,selected=winner,checkpoint=str(cp),candidates=candidates,
                            run=str(cp.parents[2].relative_to(SD)),output=str(SD/f'data/eval/ft_testscore/{run}_{winner}_test')))
    for split in ['train','validation']:
        p=SD/f'paper_reproduction/data/mit1003_fixations_{split}.json';files[str(p)]=sha(p)
    require(len(records)==14,'Wrong run count')
    write(path,dict(records=records,files=files,selection='Lowest mean rank of duration-free four-component MM-KLD and ScanMatchNoDur KLD; earlier checkpoint breaks ties; last sorts last.',sampling_seed=0,diffusion_steps=1000,limitations=['Existing validation outputs have no source-code hash; current scorer and sampler are frozen at adoption.','Image feature contents are not hashed.']))
    print('MANIFEST_SHA256',sha(path))
    for r in records:print(r['id'],r['selected'],r['candidates'][r['selected']]['scores'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--sha256');ap.add_argument('--action',choices=['build','check','test'],required=True);ap.add_argument('--index',type=int)
    a=ap.parse_args()
    if a.action=='build':build(a.manifest);return
    require(a.sha256 and sha(a.manifest)==a.sha256,'Manifest hash mismatch')
    m=json.loads(a.manifest.read_text());verify(m)
    if a.action=='check':print('PASS: 14 frozen validation-selected checkpoints');return
    require(a.index is not None and 0<=a.index<len(m['records']),'Bad index')
    import torch
    require(torch.cuda.is_available(),'GPU required')
    r=m['records'][a.index];out=Path(r['output']);require(not out.exists(),'Refusing to overwrite output')
    # Test identities are opened only inside this explicit frozen test runner.
    datafile=SD/'paper_reproduction/data/mit1003_fixations_test.json'
    data=json.loads(datafile.read_text());require(all(s['split']=='test' for s in data),'Wrong SD split labels')
    counts=Counter(s['name'] for s in data)
    canonical=SPLITS/'test.txt'
    require(canonical.exists(),'Canonical test list missing')
    keys={Path(x.strip()).stem for x in canonical.read_text().splitlines() if x.strip()}
    require(len(keys)==151 and {Path(k).stem for k in counts}==keys,'Wrong test identities')
    for split in ['train','validation']:
        other=json.loads((SD/f'paper_reproduction/data/mit1003_fixations_{split}.json').read_text())
        require(not keys & {Path(s['name']).stem for s in other},'Test overlap')
    input_hashes={str(p):sha(p) for p in [datafile,canonical]}
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,RUN=r['run'],CKPT=Path(r['checkpoint']).name,SPLIT='test')
    meta=dict(manifest_sha256=a.sha256,record=r['id'],checkpoint=r['checkpoint'],test_input_hashes=input_hashes,job=os.environ.get('SLURM_JOB_ID'))
    write(out/'frozen_started.json',meta)
    with (out/'frozen_evaluation.log').open('x') as log:
        subprocess.run(['bash',str(SD/'slurm/score_ft_valsplit.sh')],cwd=SD,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify(m)
    require(all(sha(p)==h for p,h in input_hashes.items()),'Test inputs changed')
    metrics=list(out.glob('results/metrics_epoch_*/metrics_MIT1003Dataset_test.json'))
    require(len(metrics)==1,'Missing/ambiguous final metrics')
    mm,sm=scores(metrics[0])
    for prefix in ['generations','original']:
        paths=list(out.glob(f'results/generations_epoch_*/{prefix}_MIT1003Dataset_test.pkl'))
        require(len(paths)==1,'Missing/ambiguous samples')
        with paths[0].open('rb') as f: samples=pickle.load(f)
        require(set(samples)==set(counts),'Incomplete image coverage')
        require(all(len(samples[k]['scanpaths'])==counts[k] for k in counts),'Wrong per-image path count')
    write(out/'frozen_complete.json',dict(meta,status='complete',n_images=len(counts),mm_kld_nodur=mm,sm_kld_nodur=sm,metrics_sha256=sha(metrics[0])))
    print('COMPLETE',r['id'])

if __name__=='__main__':main()
