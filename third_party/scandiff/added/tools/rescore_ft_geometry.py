"""Audit saved ScanDiff validation geometry; preserve original metric JSONs.

No dataset initialization, checkpoint loading, test reads or GPU use.
Historical transfer scoring uses a 512x384 coordinate frame and size metadata
in (height,width) order. Its upstream unpacking is correct for THAT frame.
Do not confuse this with tools/score_scanpaths.py's [width,height] inputs.
"""
import argparse
import hashlib
import io
import itertools
import json
import pickle
from pathlib import Path
import sys
from functools import lru_cache

import numpy as np
import torch
from GazeParser.ScanMatch import ScanMatch

SD = Path('/mnt/vast-nhr/projects/nim00018/tom/scandiff')
sys.path.insert(0, str(SD))
from src.gazetools.metrics.kld import perform_kld


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load(path):
    with path.open('rb') as f:
        return CPUUnpickler(f).load()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=128)
def matcher(w,h):
    return ScanMatch(Xres=w,Yres=h,Xbin=16,Ybin=12,TempBin=0)


def sequences(data, swapped=False, fixed=False):
    out = {}
    for key, entry in data.items():
        w, h = map(int, entry['size'])
        if fixed: w,h=512,384
        m = matcher(h if swapped else w,w if swapped else h)
        seqs = []
        for raw in entry['scanpaths']:
            a = np.array(raw.cpu().numpy() if torch.is_tensor(raw) else raw, copy=True)
            n = int(np.sum(a[:, 0] != -3))
            assert np.all(a[n:, 0] == -3), 'Padding must be trailing'
            xy = a[:n, :2].copy()
            xy[:, 0] *= w
            xy[:, 1] *= h
            if n < 3:
                xy = np.concatenate([xy, np.ones((3-n, 2))])
            assert np.isfinite(xy).all()
            seqs.append(m.fixationToSequence(xy).astype(int))
        out[key] = (m, seqs)
    return out


def pair_scores(m, pairs):
    """Vectorize the exact GazeParser zero-gap DP across equal-length pairs.

    Only the normalized maximum DP value is needed; traceback is irrelevant.
    Validate against GazeParser.match before using the accelerated scorer.
    """
    buckets = {}
    for a, b in pairs:
        buckets.setdefault((len(a), len(b)), []).append((a,b))
    result = []
    for (n, k), pairs in buckets.items():
        aa = np.array([a for a,b in pairs]); bb = np.array([b for a,b in pairs])
        f = np.zeros((len(pairs), n+1, k+1))
        for i in range(1,n+1):
            for j in range(1,k+1):
                f[:,i,j] = np.maximum(np.maximum(f[:,i-1,j],f[:,i,j-1]),
                    f[:,i-1,j-1]+m.SubMatrix[aa[:,i-1],bb[:,j-1]])
        vals = f.max(axis=(1,2))/(m.SubMatrix.max()*max(n,k))
        # Check independent scalar implementation for every length bucket.
        expected = m.match(pairs[0][0],pairs[0][1])[0]
        assert np.isclose(vals[0],expected,rtol=0,atol=1e-14)
        result.extend(vals.tolist())
    return result


def scores(ref, gen):
    hh, hg = [], []
    for key, (m, rr) in ref.items():
        gg = gen[key][1]
        hh.extend(pair_scores(m,itertools.combinations(rr,2)))
        hg.extend(pair_scores(m,itertools.product(rr,gg)))
    assert np.isfinite(hh).all() and np.isfinite(hg).all()
    return float(perform_kld(hh,hg)),len(hh),len(hg)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    base=SD/'data/eval/ft_valscore'
    for d in sorted(base.glob('*_val'))[:args.limit]:
        fs=list(d.glob('results/generations_epoch_*/generations_MIT1003Dataset_test.pkl'))
        assert len(fs)==1, d
        gfile=fs[0];rfile=gfile.with_name('original_MIT1003Dataset_test.pkl')
        metrics=list(d.glob('results/metrics_epoch_*/metrics_MIT1003Dataset_test.json'))
        assert len(metrics)==1,d
        metric=metrics[0]; old=json.loads(metric.read_text())
        r,g=load(rfile),load(gfile)
        assert set(r)==set(g) and len(r)==150
        assert all(tuple(r[k]['size'])==tuple(g[k]['size']) for k in r)
        upstream,hh,hg=scores(sequences(r,fixed=True),sequences(g,fixed=True))
        assert np.isclose(upstream,old['ScanMatchNoDur']['KLD'],rtol=1e-6,atol=1e-8), (d.name,upstream,old['ScanMatchNoDur']['KLD'])
        out={'protocol':'scanmatch_nodur_fixed512x384_geometry_audit_v1','split':'validation',
             'ScanMatchNoDur':{'KLD':upstream},'coordinate_frame_wh':[512,384],
             'reference_pairs':hh,'generated_reference_pairs':hg,'images':len(r),
             'inputs':{str(p):sha(p) for p in (rfile,gfile,metric)},'script_sha256':sha(Path(__file__))}
        if args.write:
            target=metric.with_name('scanmatch_nodur_geometry_audit.json')
            if target.exists():
                assert json.loads(target.read_text())==out, target
            else:
                with target.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
        print(d.name, '512x384 reproduced',upstream,flush=True)


if __name__=='__main__': main()
