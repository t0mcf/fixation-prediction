"""Freeze/check existing synthetic validation; run test only through an explicit gate.

No submissions. Existing evaluator outputs are audited, never overwritten.
DG3 test remains blocked pending the common-target-coordinate adapter review.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def verify_files(manifest):
    for p, sha in manifest['files'].items():
        require(digest(p) == sha, 'Frozen input changed: ' + p)


def check_result(path, rec, manifest, split):
    d = json.loads(Path(path).read_text())
    a, p = d['arguments'], d['provenance']
    digests = p.get('target_cells_sha256', {})
    require(all(isinstance(digests.get(g), str) and re.fullmatch('[0-9a-f]{64}', digests[g])
                for g in ('64', '224')), 'Missing or malformed target digest')
    spec = manifest['splits'][split]
    require(a['model'] == rec['family'] and a['name'] == rec['id'], 'Wrong model/name')
    require((ROOT / a['checkpoint']).resolve() == Path(rec['checkpoint']).resolve(), 'Wrong checkpoint path')
    require(p['checkpoint_sha256'] == rec['checkpoint_sha256'], 'Wrong checkpoint hash')
    require(Path(a['image_list']).resolve() == Path(spec['path']).resolve(), 'Wrong split path')
    require(p['image_list_sha256'] == manifest['files'][spec['path']], 'Wrong split hash')
    require(a['scanpath_ids'] == list(range(16)) and a['max_seq_len'] == 16, 'Wrong path selection')
    require(a['resolutions'] == [64, 224] and not a['amp'] and not p['amp'], 'Wrong grids/precision')
    require(a['parquet_path'] == manifest['parquet'] and a['imagenet_root'] == manifest['images'], 'Wrong dataset')
    require(str(p['device']).startswith('cuda'), 'Expected GPU output')
    require(p['n_images_listed'] == spec['images'], 'Wrong listed count')
    cov = p['coverage']
    require(cov['scanpath_ids'] == list(range(16)), 'Wrong coverage IDs')
    require(cov['n_scanpaths'] == spec['images'] * 16, 'Wrong scanpath count')
    if rec['family'] == 'ours':
        require(cov['length_histogram'] == {'16': spec['images'] * 16}, 'Wrong path lengths')
        require(cov['native_grid'] == 64, 'Wrong native grid')
    keys = [x.strip().split('/')[-1] for x in Path(spec['path']).read_text().splitlines() if x.strip()]
    require(len(keys) == len(set(keys)) == spec['images'], 'Ambiguous image basenames')
    for res in ('64', '224'):
        r = d['results'][res]
        rows = r['per_image']
        require(r['n_images'] == len(rows) == spec['images'], 'Wrong image count')
        require(r['n_fixations'] == spec['images'] * 240, 'Wrong target count')
        require(len({x['image'] for x in rows}) == len(rows), 'Duplicate image rows')
        require({x['image'] for x in rows} == set(keys), 'Wrong image identities')
        require(all(x['n_fixations'] == 240 for x in rows), 'Unequal/missing per-image targets')
        for short, full in [('ll', 'll_uniform'), ('nss', 'nss'), ('auc', 'auc')]:
            values = [x[short] for x in rows]
            require(all(math.isfinite(x) for x in values), 'Nonfinite score')
            avg = math.fsum(values) / len(values)
            for suffix in ('_img', '_fix'):
                require(math.isclose(avg, r[full + suffix], abs_tol=1e-7, rel_tol=1e-7), 'Aggregation mismatch')
        require(all(0 <= x['auc'] <= 1 for x in rows), 'AUC outside [0,1]')
    if split == 'validation' and rec['family'] == 'ours':
        delta = abs(d['results']['64']['ll_uniform_fix'] - rec['selection']['ll'])
        require(delta <= manifest['native_ll_tolerance'], f"Native LL reproduction failed: {rec['id']} delta={delta}")
    return digest(path)


def validation_gate(manifest):
    evidence, errors = {}, []
    reference = None
    for rec in manifest['records']:
        path = Path(manifest['validation_outputs']) / rec['id'] / 'metrics.json'
        try:
            evidence[str(path)] = check_result(path, rec, manifest, 'validation')
            digests = json.loads(path.read_text())['provenance']['target_cells_sha256']
            digests = {g: digests[g] for g in ('64', '224')}
            if reference is None:
                reference = digests
            require(digests == reference, 'Target digests differ across required runs')
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            errors.append(rec['id'] + ': ' + str(e))
    require(not errors, 'VALIDATION GATE BLOCKED\n' + '\n'.join(errors))
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--action', choices=['check', 'gate', 'seal', 'test'], required=True)
    ap.add_argument('--index', type=int)
    ap.add_argument('--receipt', type=Path)
    ap.add_argument('--receipt-sha256')
    args = ap.parse_args()
    require(digest(args.manifest) == args.manifest_sha256, 'Manifest hash mismatch')
    m = json.loads(args.manifest.read_text())
    verify_files(m)
    if args.action == 'check':
        print(f"Frozen files verified; {len(m['records'])} checkpoints; no inference")
        return
    evidence = validation_gate(m)
    if args.action == 'gate':
        print(f"VALIDATION GATE PASSED: {len(evidence)} runs, exact identities/counts, LL reproduction")
        return
    require(args.receipt is not None, '--receipt required')
    if args.action == 'seal':
        write_new(args.receipt, dict(manifest_sha256=args.manifest_sha256, evidence=evidence,
                  legacy_validation_provenance='Audited existing outputs: checkpoint/split hashes and reproduction. Original jobs did not record source hashes.'))
        print('Receipt SHA256:', digest(args.receipt))
        return
    require(args.receipt_sha256 and digest(args.receipt) == args.receipt_sha256, 'Receipt hash mismatch')
    receipt = json.loads(args.receipt.read_text())
    require(receipt['manifest_sha256'] == args.manifest_sha256 and receipt['evidence'] == evidence, 'Validation changed since seal')
    require(args.index is not None and 0 <= args.index < len(m['records']), 'Invalid record index')
    rec = m['records'][args.index]
    require(rec['family'] == 'ours', 'DG3 blocked: target-coordinate review outstanding')
    import torch
    require(torch.cuda.is_available(), 'GPU required; never infer on login node')
    out = Path(m['test_outputs']) / rec['id']
    out.mkdir(parents=True, exist_ok=False)
    result = out / 'metrics.json'
    command = [sys.executable, '-B', str(ROOT / 'scripts/eval_synthetic_fair.py'),
        '--model', rec['family'], '--checkpoint', rec['checkpoint'], '--image-list', m['splits']['test']['path'],
        '--name', rec['id'], '--out-json', str(result), '--resolutions', '64', '224',
        '--scanpath-ids', *map(str, range(16)), '--max-seq-len', '16', '--batch-size', '64', '--num-workers', '8',
        '--parquet-path', m['parquet'], '--imagenet-root', m['images']]
    meta = dict(status='running', manifest_sha256=args.manifest_sha256, receipt_sha256=args.receipt_sha256,
                command=command, job=os.environ.get('SLURM_JOB_ID'))
    write_new(out / 'started.json', meta)
    env = dict(os.environ, SYNTHETIC_TEST_OK='1', OURS_CROP_TRANSFORM='1', PYTHONDONTWRITEBYTECODE='1')
    with (out / 'evaluation.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_files(m)
    require(validation_gate(m) == evidence, 'Validation mutated during test')
    sha = check_result(result, rec, m, 'test')
    write_new(out / 'provenance.json', dict(meta, status='complete', result_sha256=sha))
    print('Completed:', rec['id'])


if __name__ == '__main__':
    main()
