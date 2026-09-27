"""Extend the audited v2 manifest without opening test lists or running inference."""
import argparse
import csv
import json
from pathlib import Path
import torch
from run_frozen_indomain_v3 import digest, require, write_new

ROOT = Path(__file__).resolve().parents[1]
DG = Path('/user/tomcosmo.fischer/u27846/repos/scanpather').resolve()
BASE_SHA = 'af834cc3d8a2b678a6753d7e142b7ab10c21c3f445f6e05360e894ff886210a5'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    base = ROOT/'docs/report_tables/indomain_test_manifest_20260912_v2.json'
    require(digest(base) == BASE_SHA, 'v2 changed')
    m = json.loads(base.read_text())
    m['schema'] = 'frozen_indomain_v3'
    m['parent_manifest_sha256'] = BASE_SHA
    # Keep all old frozen data/checkpoint/split hashes. Only the reviewed evaluator
    # is refreshed; any other drift remains a failure in the frozen runner.
    ev = str((ROOT/'scripts/eval_synthetic_fair.py').resolve())
    m['source_update'] = {ev: {'previous': m['files'][ev], 'current': digest(ev)}}
    m['files'][ev] = digest(ev)
    for name in ['prepare_indomain_v3.py','run_frozen_indomain_v3.py','check_target_identity.py']:
        p = ROOT/'scripts'/name
        m['files'][str(p)] = digest(p)
    p = ROOT/'slurm/frozen_indomain_v3.sbatch'
    m['files'][str(p)] = digest(p)
    # Freeze the local DG3 adapter dependencies and model implementation too.
    for folder in [DG/'scanpath', DG/'models/DeepGaze/deepgaze_pytorch']:
        for p in folder.rglob('*.py'):
            m['files'][str(p)] = digest(p)
    for size in ['1k','10k','50k','100k','200k']:
        run = DG/f'runs/imagenet_paths/dg3_{size}_5sp/scanpath'
        cp, cfg, log = run/'best.pth', run/'config.json', run/'log.csv'
        rows = list(csv.DictReader(log.open()))
        best = max(rows, key=lambda r: float(r['validation_LL']))
        state = torch.load(cp, map_location='cpu', weights_only=False)
        require(int(state['step']) == int(best['epoch']), 'DG3 checkpoint is not best validation LL: '+size)
        settings = json.loads(cfg.read_text())['config']['training']
        require(settings['stage2']['downsample'] == 1, 'DG3 grid setting changed')
        require(not settings['centerbias']['use_centerbias'], 'Unexpected DG3 center bias')
        for p in [cp,cfg,log]:
            m['files'][str(p)] = digest(p)
        m['records'].append(dict(id='dg3_'+size, family='dg3', checkpoint=str(cp),
            checkpoint_sha256=m['files'][str(cp)], config_json=str(cfg), config_sha256=m['files'][str(cfg)],
            selection=dict(criterion='maximum training-validation LL', step=int(state['step']),
                ll=float(best['validation_LL']), completed_epochs=max(int(r['epoch']) for r in rows),
                epoch_budget=settings['stage2']['max_epochs'], budget_limited=(size=='200k'))))
        del state
    require(len(m['records']) == 20, 'Wrong run count')
    m['limitations'] = [x for x in m['limitations'] if not x.startswith('DG3 excluded')]
    m['limitations'].append('DG3 200k completed two of ten scanpath-stage epochs.')
    m['selection_policy'] = 'Ours and DG3: best validation LL for conditional and whole-scanpath evaluation, per supervisor decision.'
    write_new(args.output,m)
    print('Manifest SHA256:', digest(args.output), flush=True)

if __name__ == '__main__': main()
