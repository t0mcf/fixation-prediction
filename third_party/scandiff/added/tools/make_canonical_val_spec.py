"""
Build a canonical-val ground-truth spec in the same format score_scanpaths.py
consumes for MIT1003, so EVERY model (ours / DeepGaze III / ScanDiff) can be
sampled and scored on the synthetic in-domain validation set through the exact
same path.

Why this set: the MIT1003 realism curves answer a TRANSFER question ("do the
generated scanpaths look human?"). The in-domain scaling question ("does more
synthetic training data make the model better at the distribution it is trained
on?") needs the held-out synthetic canonical val set instead.

Note: every canonical-val scanpath is exactly 16 fixations, so the length
oracle is trivial here and the length-matching artefact that complicated the
MIT1003 comparison cannot arise.

Outputs:
  <out-dir>/canonical_val_gt.json    {stem: {size:[W,H], scanpaths:[{X,Y,T,subject}]}}
  <out-dir>/stimuli/<stem>.JPEG      flat symlinks (samplers expect a flat dir)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SRC = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/canonical_val")
IMAGENET = Path("/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixations", default=str(SRC / "fixations_validation.json"))
    ap.add_argument("--out-dir", default="/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval/canonical_val")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "stimuli").mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.fixations))
    by_img: dict[str, list] = defaultdict(list)
    sizes: dict[str, list] = {}
    for r in rows:
        stem = Path(r["name"]).stem
        sizes[stem] = [int(r["width"]), int(r["height"])]
        by_img[stem].append({
            "X": list(map(float, r["X"])),
            "Y": list(map(float, r["Y"])),
            "T": list(map(float, r.get("T", [100.0] * len(r["X"])))),
            "subject": int(r.get("subject", 0)),
        })

    spec = {stem: {"size": sizes[stem], "scanpaths": sps}
            for stem, sps in sorted(by_img.items())}
    spec_path = out / "canonical_val_gt.json"
    json.dump(spec, open(spec_path, "w"))

    n_link, n_missing = 0, 0
    for stem in spec:
        cls = stem.split("_")[0]
        src = IMAGENET / cls / f"{stem}.JPEG"
        dst = out / "stimuli" / f"{stem}.JPEG"
        if not src.exists():
            n_missing += 1
            continue
        if not dst.exists():
            dst.symlink_to(src)
        n_link += 1

    lens = [len(sp["X"]) for v in spec.values() for sp in v["scanpaths"]]
    print(f"wrote {spec_path}")
    print(f"  images: {len(spec)} | scanpaths: {len(lens)} | "
          f"lengths min/max: {min(lens)}/{max(lens)}")
    print(f"  stimuli symlinks: {n_link} ({n_missing} missing)")


if __name__ == "__main__":
    main()
