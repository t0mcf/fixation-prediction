"""Score the fair harness's centre-bias prior itself (AUC / NSS / LL) on the
MIT1003 VALIDATION split, so its strength can be compared with the
dataset-level centre bias that DeepGaze3.5-VL reports (AUC 0.81 on
MIT1003, Table 1). Written 2026-09-17 for the caption of the headline
variant with the external reference; no evaluation code is changed; the
prior is fitted exactly as scripts/eval_mit1003_fair.py does (tuned mode,
train.txt fixations) and scored with the same metric functions on the same
target set (first 15 real fixations per scanpath, image-averaged and
fixation-weighted). Refuses the test split.

  python scripts/centerbias_reference.py --grid 224
  -> docs/report_tables/centerbias_reference.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))

from eval_mit1003_fair import fixed_split, train_fold_fixations_by_image  # noqa: E402
from src.data.mit1003_dataset import MIT1003Dataset  # noqa: E402
from src.eval.fair_metrics import fit_centerbias_tuned, coords_to_cells  # noqa: E402
from src.training.metrics import nss_score, auc_score  # noqa: E402

SPLIT_DIR = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003")
OUT = root / "docs/report_tables/centerbias_reference.csv"


def target_fixations_by_image(images):
    """Same target set as the harness: per scanpath the first min(n_valid, 15)
    real fixations (the model predicts fixations[:, 1:] with max_seq_len 16)."""
    ds = MIT1003Dataset(max_seq_len=16, min_fixations=1, image_indices=images)
    by_img = {}
    for rec in ds._records:
        H, W = ds._img_shapes[rec["img_idx"]]
        n = min(rec["n_valid"], 15)
        x = np.clip((rec["xs"][:n] / (W - 1)) * 2.0 - 1.0, -1.0, 1.0)
        y = np.clip((rec["ys"][:n] / (H - 1)) * 2.0 - 1.0, -1.0, 1.0)
        by_img.setdefault(rec["img_idx"], []).append(
            np.stack([x, y], axis=1).astype(np.float32))
    return {k: torch.from_numpy(np.concatenate(v, 0)) for k, v in by_img.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=224)
    ap.add_argument("--eval-split", default="validation", choices=["validation"])
    args = ap.parse_args()

    (_, train_images, val_images), = list(fixed_split(SPLIT_DIR, args.eval_split))
    cb_per_img = train_fold_fixations_by_image(train_images)
    cb = fit_centerbias_tuned(cb_per_img, args.grid, device="cpu", verbose=True)  # (g,g) log-density
    cb = cb.float()
    log_norm = math.log(args.grid * args.grid)

    targets = target_fixations_by_image(val_images)
    per_img = []   # (n, mean_ll_bits_over_uniform, nss, auc)
    for iid, txy in targets.items():
        n = len(txy)
        row, col = coords_to_cells(txy, args.grid)
        ll = (cb[row, col] + log_norm) / math.log(2.0)
        maps = cb.unsqueeze(0).expand(n, -1, -1)
        per_img.append((n, float(ll.mean()), float(nss_score(maps, txy)),
                        float(auc_score(maps, txy))))
    n_tot = sum(p[0] for p in per_img)
    img = {"ll": np.mean([p[1] for p in per_img]), "nss": np.mean([p[2] for p in per_img]),
           "auc": np.mean([p[3] for p in per_img])}
    fix = {k: sum(p[i] * p[0] for p in per_img) / n_tot
           for k, i in (("ll", 1), ("nss", 2), ("auc", 3))}
    row = {"split": args.eval_split, "grid": args.grid, "cb_mode": "tuned (train.txt fixations)",
           "n_images": len(per_img), "n_fixations": n_tot,
           "cb_ll_uniform_img": round(img["ll"], 4), "cb_ll_uniform_fix": round(fix["ll"], 4),
           "cb_nss_img": round(img["nss"], 4), "cb_nss_fix": round(fix["nss"], 4),
           "cb_auc_img": round(img["auc"], 4), "cb_auc_fix": round(fix["auc"], 4)}
    print(row)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader(); w.writerow(row)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
