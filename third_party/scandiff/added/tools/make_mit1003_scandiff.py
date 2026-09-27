"""
Build MIT1003 as a ScanDiff TEST dataset (for evaluating models on real human gaze).

Outputs under {out_dir} (default data/mit1003):
  mit1003_fixations_test.json        (all scanpaths, split="test")
  mit1003_fixations_train.json       (empty — we only use MIT1003 for eval)
  mit1003_fixations_validation.json  (empty)
  images_list.txt                    (<abs stimulus path>\t<stem>) for feature extraction

Each test entry: {name:"<stem>.jpeg", split:"test", width:W, height:H,
                  X:[px], Y:[px], T:[ms], subject:id}.
Durations come from scanpath_fixation_attributes/durations (seconds -> *1000 = ms);
they only affect with-duration metrics, which we don't report.
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
    ap.add_argument("--out-dir", default="data/mit1003")
    ap.add_argument("--min-fix", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with h5py.File(root / "fixations.hdf5", "r") as f:
        xs = f["train_xs"][:]; ys = f["train_ys"][:]; ns = f["train_ns"][:]
        subj = f["train_subjects"][:]
        durs = f["scanpath_fixation_attributes"]["durations"][:]
    with h5py.File(root / "stimuli.hdf5", "r") as f:
        filenames = [fn.decode() if isinstance(fn, bytes) else fn for fn in f["filenames"][:]]
        shapes = f["shapes"][:]   # (n_img, 3) = (H, W, C)

    test = []
    imglist = {}
    kept = dropped = 0
    for i in range(len(xs)):
        valid = ~np.isnan(xs[i])
        n = int(valid.sum())
        if n < args.min_fix:
            dropped += 1; continue
        img_idx = int(ns[i])
        stem = Path(filenames[img_idx]).stem
        H, W = int(shapes[img_idx][0]), int(shapes[img_idx][1])
        test.append({
            "name": f"{stem}.jpeg", "split": "test", "width": W, "height": H,
            "X": xs[i][valid].astype(float).tolist(),
            "Y": ys[i][valid].astype(float).tolist(),
            "T": (np.nan_to_num(durs[i][valid]) * 1000.0).astype(float).tolist(),
            "subject": int(subj[i]),
        })
        imglist[stem] = str(root / filenames[img_idx])
        kept += 1

    json.dump(test, open(out / "mit1003_fixations_test.json", "w"))
    json.dump([], open(out / "mit1003_fixations_train.json", "w"))
    json.dump([], open(out / "mit1003_fixations_validation.json", "w"))
    with open(out / "images_list.txt", "w") as f:
        for stem, src in imglist.items():
            f.write(f"{src}\t{stem}\n")
    print(f"MIT1003 test: {kept} scanpaths over {len(imglist)} images "
          f"(dropped {dropped} <{args.min_fix} fix) -> {out}")


if __name__ == "__main__":
    main()
