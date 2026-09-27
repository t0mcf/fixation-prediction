"""Prepare a new, immutable-by-hash ours manifest; no inference or submission."""
import argparse
import csv
import json
from pathlib import Path
import torch
from run_frozen_indomain import digest, require, write_new

ROOT = Path(__file__).resolve().parents[1]
SPLITS = Path('/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), 'Manifest already exists')
    files = {}
    for folder in [ROOT / 'src', ROOT / 'scripts']:
        for p in folder.rglob('*.py'):
            # Only runtime modules below src; evaluator and builder scripts below scripts.
            if folder.name == 'scripts' and p.name not in ['eval_synthetic_fair.py', 'evaluate_checkpoint.py', 'run_frozen_indomain.py', 'prepare_indomain_test.py']:
                continue
            files[str(p)] = digest(p)
    launcher = ROOT / 'slurm/frozen_indomain.sbatch'
    files[str(launcher)] = digest(launcher)
    sets = {}
    for p in SPLITS.glob('*.txt'):
        files[str(p)] = digest(p)
        sets[p.stem] = set(p.read_text().splitlines())
    require(len(sets['validation_5k']) == 5000 and len(sets['test_10k']) == 10000, 'Split sizes')
    require(not sets['validation_5k'] & sets['test_10k'], 'Split overlap')
    require(sets['scanpath_ids_evaluation'] == set(map(str, range(16))), 'Evaluation IDs changed')
    records = []
    for seed, sizes in [(42, ['1k','10k','50k','100k','200k','400k','800k']),
                        (43, ['1k','10k','50k','100k','200k']), (44, ['1k','10k','50k'])]:
        for size in sizes:
            run = ROOT / f'runs/v2_{size}_5p_ll_seed{seed}'
            cp, cfg, log = run/'checkpoints/best_val_ll.pt', run/'config.json', run/'log.csv'
            c = json.loads(cfg.read_text())
            train_list = Path(c['train_image_list'])
            require(train_list == SPLITS/f'train_{size}.txt', 'Unexpected training list')
            train = set(train_list.read_text().splitlines())
            require(not train & (sets['validation_5k'] | sets['test_10k']), 'Training overlap')
            require(c['train_scanpath_ids'] == list(range(5)) and c['ours_crop_transform'], 'Training protocol mismatch')
            rows = [r for r in csv.DictReader(log.open()) if r['split'] == 'val']
            require(float(rows[-1]['epoch']) == 18, 'Incomplete run')
            best = max(rows, key=lambda r: float(r['ll']))
            state = torch.load(cp, map_location='cpu', weights_only=False)
            require(state['step'] == int(best['step']) and state['epoch'] + 1 == int(float(best['epoch'])), 'Checkpoint not best LL step')
            require(state['args']['max_seq_len'] == 16 and state['args']['heatmap_size'] == 64, 'Architecture mismatch')
            for p in [cp, cfg, log]: files[str(p)] = digest(p)
            records.append(dict(id=f'ours_{size}_seed{seed}', family='ours', checkpoint=str(cp),
                checkpoint_sha256=files[str(cp)], selection=dict(epoch=float(best['epoch']), step=int(best['step']), ll=float(best['ll']))))
            del state
    parquet = Path('/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet')
    # Full content hash: expensive once, but no fixation decoding or test inference.
    files[str(parquet)] = digest(parquet)
    manifest = dict(schema='frozen_indomain_v1', records=records, files=files,
        parquet=str(parquet), images='/mnt/vast-nhr/projects/nim00018/datasets/ImageNet',
        native_ll_tolerance=0.005, tolerance_policy='Conservative maximum: failure requires investigation, never automatic widening.',
        splits={s:dict(path=str(SPLITS/f'{stem}.txt'), images=n) for s,stem,n in [('validation','validation_5k',5000),('test','test_10k',10000)]},
        validation_outputs=str(ROOT/'runs/synthetic_fair/validation'),
        test_outputs=str(ROOT/'runs/synthetic_fair/test'),
        limitations=['Image-file contents are not hashed by this manifest.',
                      'Existing validation jobs do not carry source-code hashes; adoption is explicit.',
                      'DG3 excluded pending common target-coordinate review.'])
    write_new(args.output, manifest)
    print('Manifest SHA256:', digest(args.output), flush=True)


if __name__ == '__main__': main()
