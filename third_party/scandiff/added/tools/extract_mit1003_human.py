"""
Extract human MIT1003 scanpaths into the dict format ScanDiff's metrics expect.

Reads raw MIT1003 (fixations.hdf5 + stimuli.hdf5) and produces, per stimulus:
  { "<stem>": { "size": [W, H],
                "scanpaths": [ {"X":[...], "Y":[...], "T":[...ms...], "subject": s}, ... ] } }

'size' is (W, H) to match ScanDiff's dataset convention (original_img_size = (width,height)).
Coordinates are absolute pixels; durations in ms (from scanpath_fixation_attributes/durations).
Saved as JSON. Optionally limit to the first N images (--n-images) for quick runs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py
import numpy as np

MIT = Path("/mnt/vast-nhr/projects/nim00018/datasets/MIT1003")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(MIT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=None, help="limit to first N images")
    ap.add_argument("--min-fix", type=int, default=3, help="drop scanpaths shorter than this")
    args = ap.parse_args()
    root = Path(args.root)

    with h5py.File(root / "fixations.hdf5", "r") as f:
        xs = f["train_xs"][:]            # (N, 15) pixel x, NaN-padded
        ys = f["train_ys"][:]            # (N, 15)
        ns = f["train_ns"][:]            # (N,) image index
        subj = f["train_subjects"][:]    # (N,)
        durs = f["scanpath_fixation_attributes"]["durations"][:]   # (N, 15) ms

    with h5py.File(root / "stimuli.hdf5", "r") as f:
        filenames = [fn.decode() if isinstance(fn, bytes) else fn for fn in f["filenames"][:]]
        shapes = f["shapes"][:]          # (n_img, 3) = (H, W, C)

    # which image indices to keep
    keep_imgs = set(range(len(filenames)))
    if args.n_images:
        keep_imgs = set(range(min(args.n_images, len(filenames))))

    data: dict[str, dict] = {}
    kept = dropped = 0
    for i in range(len(xs)):
        img_idx = int(ns[i])
        if img_idx not in keep_imgs:
            continue
        x_row, y_row, d_row = xs[i], ys[i], durs[i]
        valid = ~np.isnan(x_row)
        n = int(valid.sum())
        if n < args.min_fix:
            dropped += 1
            continue
        stem = Path(filenames[img_idx]).stem
        H, W = int(shapes[img_idx][0]), int(shapes[img_idx][1])
        entry = data.setdefault(stem, {"size": [W, H], "scanpaths": []})
        entry["scanpaths"].append({
            "X": x_row[valid].astype(float).tolist(),
            "Y": y_row[valid].astype(float).tolist(),
            "T": np.nan_to_num(d_row[valid]).astype(float).tolist(),
            "subject": int(subj[i]),
        })
        kept += 1

    with open(args.out, "w") as f:
        json.dump(data, f)
    n_sp = sum(len(v["scanpaths"]) for v in data.values())
    print(f"images: {len(data)} | scanpaths kept: {kept} (dropped {dropped} < {args.min_fix} fix)")
    print(f"avg scanpaths/image: {n_sp/max(len(data),1):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
