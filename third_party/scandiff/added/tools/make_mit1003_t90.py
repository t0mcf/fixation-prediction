"""
Build data/mit1003_t90: MIT1003 in ScanDiff training format with a 90/10
image split for Phase 1 of the ScanDiff-as-data-generator experiment.

Held-out 10% = fixation-prediction's seed-42 fold-0 VAL images (101 images),
so transfer evals on fold 0 are provably leak-free w.r.t. this teacher.
The remaining 902 images become ScanDiff's train split; the held-out images
double as validation and test.

Sources: data/mit1003 (eval-only build: all scanpaths in the 'test' json,
real durations in ms) + stimuli.hdf5 order (== pysaliency order, verified).
"""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

SD = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SRC = SD / "data" / "mit1003"
DST = SD / "data" / "mit1003_t90"
MIT = Path("/mnt/vast-nhr/projects/nim00018/datasets/MIT1003")

# seed-42 fold-0 split, byte-identical to train.py / eval_mit1003_fair.py
perm = np.random.default_rng(42).permutation(1003)
val_idx = set(np.array_split(perm, 10)[0].tolist())

with h5py.File(MIT / "stimuli.hdf5") as f:
    stems = [Path(fn.decode() if isinstance(fn, bytes) else fn).stem
             for fn in f["filenames"][:]]
val_stems = {stems[i] for i in val_idx}

data = json.load(open(SRC / "mit1003_fixations_test.json"))
train, heldout = [], []
for rec in data:
    stem = Path(rec["name"]).stem
    if stem in val_stems:
        heldout.append(rec)
    else:
        train.append(rec)

DST.mkdir(parents=True, exist_ok=True)
for split, recs in [("train", train), ("validation", heldout), ("test", heldout)]:
    out = []
    for r in recs:
        r = dict(r)
        r["split"] = split
        out.append(r)
    with open(DST / f"mit1003_t90_fixations_{split}.json", "w") as f:
        json.dump(out, f)
    print(f"{split}: {len(out)} scanpaths / {len(set(r['name'] for r in out))} images")

# features + clusters: same images, reuse via symlink/copy
feat = DST / "dinov2_base_timm_image_features"
if not feat.exists():
    feat.symlink_to("../mit1003/dinov2_base_timm_image_features")
clusters_src = SRC / "clusters_mit1003_512_384.npy"
clusters_dst = DST / "clusters_mit1003_t90_512_384.npy"
if not clusters_dst.exists():
    shutil.copy(clusters_src, clusters_dst)
print("features symlinked, clusters copied ->", DST)
