"""Build a ScanDiff-format validation dataset from the actual canonical
5000-image val split (same one used for NSS/AUC/LL across ours/DG3/ceiling),
instead of ScanDiff's own disjoint per-scale validation jsons.

Picks a random subset of N canonical images (fixed seed), writes:
  - a fixations validation json in ScanDiff's schema (name, width, height,
    X/Y in absolute pixels, T in ms, split, subject)
  - images_list.txt for tools/extract_dinov2_features.py

Usage:
  python scripts/build_canonical_val_dataset.py --n-images 100
  # then:
  python tools/extract_dinov2_features.py \
    --images-list runs_ours/canonical_val/images_list.txt \
    --out-dir runs_ours/canonical_val/features --batch-size 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PARQUET = "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
IMAGENET_ROOT = Path("/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images")


def _canonical_key(p: str) -> str:
    parts = p.split("/")
    return parts[-2] + "/" + parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs_ours/canonical_val")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET, columns=["image_path", "locations"])
    keys = df["image_path"].map(_canonical_key)
    unique_keys = sorted(keys.unique())
    rng = np.random.default_rng(3141)
    canonical_val_keys = [unique_keys[i] for i in rng.permutation(len(unique_keys))[:5000]]

    is_val = keys.isin(set(canonical_val_keys))
    val_df = df[is_val].reset_index(drop=True)
    val_df["key"] = keys[is_val].values

    subset_rng = np.random.default_rng(args.seed)
    chosen_keys = subset_rng.choice(canonical_val_keys, size=min(args.n_images, len(canonical_val_keys)), replace=False)
    print(f"canonical val pool: {len(canonical_val_keys)} images; chosen subset: {len(chosen_keys)}")

    fixations = []
    images_list_lines = []
    missing = 0
    for key in chosen_keys:
        rows = val_df[val_df["key"] == key]
        if len(rows) == 0:
            missing += 1
            continue
        stem_with_ext = key.split("/")[-1]  # e.g. n01440764_10108.JPEG
        wnid = stem_with_ext.split("_")[0]
        img_path = IMAGENET_ROOT / wnid / stem_with_ext
        if not img_path.exists():
            missing += 1
            continue
        with Image.open(img_path) as im:
            width, height = im.size

        stem = Path(stem_with_ext).stem  # n01440764_10108 (no extension)
        # name field: lowercase extension to match oursynth's own convention
        name = f"{stem}.jpeg"

        for subj_idx, (_, row) in enumerate(rows.iterrows()):
            locs = np.array(row["locations"], dtype=np.float32).reshape(-1, 2)
            px = ((locs[:, 0] + 1.0) / 2.0 * width).tolist()
            py = ((locs[:, 1] + 1.0) / 2.0 * height).tolist()
            durations = [200.0] * len(px)
            fixations.append({
                "name": name, "img_filename": name, "split": "validation",
                "width": width, "height": height,
                "X": px, "Y": py, "T": durations, "subject": subj_idx,
            })

        images_list_lines.append(f"{img_path}\t{stem}")

    print(f"built {len(fixations)} fixation rows over {len(images_list_lines)} images ({missing} missing/skipped)")

    with open(out_dir / "fixations_validation.json", "w") as f:
        json.dump(fixations, f)
    with open(out_dir / "images_list.txt", "w") as f:
        f.write("\n".join(images_list_lines) + "\n")

    print(f"saved {out_dir/'fixations_validation.json'} and {out_dir/'images_list.txt'}")


if __name__ == "__main__":
    main()
