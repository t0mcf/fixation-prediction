"""Freeze six source-comparison fine-tunes, using validation only."""
import csv
import json
from pathlib import Path
import torch
from run_frozen_indomain_v3 import digest, require, write_new

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT/'docs/report_tables/transfer_test_manifest_ours_20260908.json').read_text())
deps = {k:v for k,v in base['dependencies'].items() if k.startswith(base['split_dir']+'/')}
# Reuse already frozen split hashes, with no test-list reads in this builder.
for p in (ROOT/'src').rglob('*.py'):
    deps[str(p)] = digest(p)
for name in ['eval_mit1003_fair.py','run_frozen_transfer.py','prepare_triangle_test.py']:
    p = ROOT/'scripts'/name; deps[str(p)] = digest(p)
records=[]
for arm in ['sd5','avfull','avmatch']:
    for seed in [42,43]:
        run=ROOT/f'runs/ft_authsplit_{arm}_seed{seed}'
        cfg=json.loads((run/'config.json').read_text())
        require(cfg['dataset']=='mit1003' and cfg['mit_split_dir']==base['split_dir'], 'Wrong dataset/splits')
        require(cfg['seed']==seed and cfg['num_epochs']==30 and cfg['val_batches']==-1, 'Wrong training protocol')
        require(not cfg.get('ours_crop_transform',False), 'Wrong human preprocessing')
        expected=ROOT/f'runs/triangle_{arm}_v2u_100k/checkpoints/final.pt'
        require((ROOT/cfg['init_from']).resolve()==expected.resolve(), 'Wrong source initialization')
        rows=[r for r in csv.DictReader((run/'log.csv').open()) if r['split']=='val']
        require(len(rows)==30 and float(rows[-1]['epoch'])==30, 'Incomplete training')
        winner=max(rows,key=lambda r:float(r['ll']))
        cp=run/'checkpoints/best_val_ll.pt'
        ck=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
        require(ck['epoch']+1==int(float(winner['epoch'])) and ck['step']==int(winner['step']), 'Wrong selected checkpoint')
        records.append(dict(id=f'ours_{arm}_seed{seed}',family='ours',condition=arm,replicate=seed,run=str(run),checkpoint=str(cp),checkpoint_sha256=digest(cp),selected_epoch=ck['epoch']+1,validation_ll=float(winner['ll']),selection='maximum training-validation LL; first tie',init_path=str(expected)))
        for p in [run/'config.json',run/'log.csv',expected]:deps[str(p)]=digest(p)
        del ck
m=dict(schema_version=1,scope='Six fixed-split MIT1003 source-comparison fine-tunes; scratch reused, shuffled excluded',split_dir=base['split_dir'],resolution=224,records=records,dependencies=deps)
p=ROOT/'docs/report_tables/triangle_test_manifest_20260912.json';write_new(p,m)
print('Manifest SHA256:',digest(p))
