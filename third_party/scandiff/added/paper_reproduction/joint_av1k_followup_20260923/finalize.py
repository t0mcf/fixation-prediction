from pathlib import Path
import os, sys, json, hashlib, subprocess, importlib.util
import yaml
S=Path('/mnt/vast-nhr/projects/nim00018/tom/scandiff')
R=Path.home()/'repos/fixation-prediction'
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen', R/'scripts/frozen_scandiff_transfer.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
p=json.loads((E/'protocol.json').read_text())
for path,h in p['files'].items():
    f.require(f.sha(path)==h, 'Protocol file changed: '+path)
if '--check' in sys.argv:
    print('PROTOCOL VERIFIED');sys.exit(0)
files=dict(p['files']); records=[]
for seed in [0,1]:
    name=f'armB_joint_av1k_seed{seed}_bvinit'
    run='paper_reproduction/'+name
    candidates={}
    traincfg=S/run/'train/.hydra/config.yaml'
    cfg=yaml.safe_load(traincfg.read_text())
    f.require(cfg['trainer']['max_epochs']==200 and cfg['seed']==seed,'Training settings changed')
    f.require(Path(cfg['init_from'])==Path(p['initial_checkpoint']),'Wrong initialization')
    files[str(traincfg)]=f.sha(traincfg)
    for label in p['validation_grid']:
        d=S/'data/eval/ft_valscore'/f'{name}_{label}_val'
        met=list(d.glob('results/metrics_epoch_*/metrics_MIT1003Dataset_test.json'))
        f.require(len(met)==1,'Missing validation '+str(d))
        cpath=d/'hydra/.hydra/config.yaml'; c=yaml.safe_load(cpath.read_text())
        cp=S/run/'train/checkpoints'/f'{label}.pt'
        f.require(c['data']['test_datasets']['mit1003']['split']=='valid','Wrong split')
        f.require(Path(c['ckpt_path'])==cp and c['seed']==0 and c['diffusion']['num_timesteps']==1000,'Wrong evaluation settings')
        candidates[label]=dict(scores=list(f.scores(met[0])),metrics=str(met[0]),config=str(cpath),checkpoint=str(cp))
        for q in [met[0],cpath]: files[str(q)]=f.sha(q)
    winner=f.choose(candidates);cp=Path(candidates[winner]['checkpoint']);files[str(cp)]=f.sha(cp)
    records.append(dict(id=name,selected=winner,checkpoint=str(cp),candidates=candidates,run=run,output=str(S/'data/eval/ft_testscore'/f'{name}_{winner}_test')))
# Freeze both sampling draws before any test evaluation.
manifest_paths=[]
for draw in [0,1]:
    rec=json.loads(json.dumps(records))
    if draw:
        for row in rec: row['output']+='_s1'
    m=dict(records=rec,files=files,selection=p['selection'],sampling_seed=draw,diffusion_steps=1000)
    dest=E/f'test_manifest_draw{draw}.json'; f.write(dest,m)
    manifest_paths.append((dest,f.sha(dest)))
f.write(E/'manifest_hashes.json',{str(q):h for q,h in manifest_paths})
for draw,(manifest,h) in enumerate(manifest_paths):
    for i,row in enumerate(records):
        env=dict(os.environ,GEN_SEED=str(draw),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,'-B',str(R/'scripts/frozen_scandiff_transfer.py'),'--manifest',str(manifest),'--sha256',h,'--action','test','--index',str(i)],env=env,cwd=S,check=True)
        suffix='_s1' if draw else ''
        out=Path(row['output']+suffix)
        paths=list(out.glob('results/generations_epoch_*/generations_MIT1003Dataset_test.pkl'))
        f.require(len(paths)==1,'Missing generations')
        label=f"scandiff_ft_{row['id']}_{row['selected']}{suffix}"
        dest=E/label;dest.mkdir(exist_ok=False)
        subprocess.run([sys.executable,'tools/scandiff_gen_to_npy.py','--gen-pkl',str(paths[0]),'--out',str(dest/'gen.npy')],cwd=S,check=True)
        subprocess.run([sys.executable,'tools/score_scanpaths.py','--human',str(S/'data/eval/mit1003_human_initial_test.json'),'--generated',str(dest/'gen.npy'),'--label',label,'--csv',str(E/'xfam_test_scores.csv')],cwd=S,check=True)
f.write(E/'complete.json',dict(status='complete',manifest_hashes={str(q):h for q,h in manifest_paths},note='New condition only; existing global aggregates untouched.'))
