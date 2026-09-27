"""
Generalized version of make_mit1003_t90.py: build data/mit1003_fold{K} for any
CV fold K (not just fold 0), same seed-42/10-fold split as
fixation-prediction's src/data/dataset.py and the CV fairness campaign.

Why this exists: the model-vs-metric-dependence investigation (2026-08-07,
engineering_log.md 19.4b/c) is currently single-fold (fold 0 only, using
mit1003_t90). To make it a citable multi-fold result, ScanDiff needs
scratch-on-human and pretrained-then-finetuned checkpoints for additional
folds too -- this script builds the training data for those folds.

Usage:
  python tools/make_mit1003_fold.py --fold 1
  python tools/make_mit1003_fold.py --fold 2
  python tools/make_mit1003_fold.py --fold 3
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

SD = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SRC = SD / "data" / "mit1003"
MIT = Path("/mnt/vast-nhr/projects/nim00018/datasets/MIT1003")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="CV fold index (0-9)")
    ap.add_argument("--num-folds", type=int, default=10)
    args = ap.parse_args()

    DST = SD / "data" / f"mit1003_fold{args.fold}"

    # seed-42 fold split, byte-identical to train.py / eval_mit1003_fair.py /
    # make_mit1003_t90.py -- do not change this logic.
    perm = np.random.default_rng(42).permutation(1003)
    val_idx = set(np.array_split(perm, args.num_folds)[args.fold].tolist())

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
        with open(DST / f"mit1003_fold{args.fold}_fixations_{split}.json", "w") as f:
            json.dump(out, f)
        print(f"{split}: {len(out)} scanpaths / {len(set(r['name'] for r in out))} images")

    feat = DST / "dinov2_base_timm_image_features"
    if not feat.exists():
        feat.symlink_to("../mit1003/dinov2_base_timm_image_features")
    clusters_src = SRC / "clusters_mit1003_512_384.npy"
    clusters_dst = DST / f"clusters_mit1003_fold{args.fold}_512_384.npy"
    if not clusters_dst.exists():
        shutil.copy(clusters_src, clusters_dst)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
