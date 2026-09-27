"""Create ScanDiff-format dataset JSONs from the same parquet used by our model.

Replicates the exact image selection and val split from
fixation-prediction/src/data/dataset.py (canonical seed=3141, 5000 val images)
so that ScanDiff trains on identical data.

Usage:
    python tools/make_scandiff_dataset.py \
        --out-dir ./data/oursynth_200k \
        --max-images 200000 \
        --max-paths-per-image 5
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PARQUET = ("/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
           "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet")
VAL_SEED = 3141
VAL_N = 5000


def canonical_split(df):
    """Exact replica of ScanpathDataset's canonical val split (seed=3141)."""
    def _canonical_key(p):
        parts = p.split('/')
        return parts[-2] + '/' + parts[-1]

    keys = df["image_path"].map(_canonical_key)
    unique_keys = sorted(keys.unique())
    rng = np.random.default_rng(VAL_SEED)
    val_keys = set(
        unique_keys[i]
        for i in rng.permutation(len(unique_keys))[:VAL_N]
    )
    is_val = keys.isin(val_keys)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def to_entries(df, split_label, canvas, dummy_dur):
    entries = []
    for row in df.itertuples(index=False):
        loc = np.asarray(row.locations, dtype=np.float64).reshape(-1, 2)
        n_fix = len(np.asarray(row.log_pis)) + 1
        loc = loc[:n_fix]
        if n_fix < 3:
            continue
        xy01 = np.clip((loc + 1.0) / 2.0, 0.0, 1.0)
        X = (xy01[:, 0] * canvas).tolist()
        Y = (xy01[:, 1] * canvas).tolist()
        stem = Path(row.image_path).stem
        entries.append({
            "name": f"{stem}.jpeg",
            "img_filename": f"{stem}.jpeg",
            "split": split_label,
            "width": canvas,
            "height": canvas,
            "X": X, "Y": Y,
            "T": [dummy_dur] * n_fix,
            "subject": int(row.epoch),
        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--max-images", type=int, default=None,
                    help="number of training images (matches our model's max_images)")
    ap.add_argument("--max-paths-per-image", type=int, default=5)
    ap.add_argument("--canvas", type=int, default=1000)
    ap.add_argument("--dummy-duration-ms", type=float, default=200.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet, engine="pyarrow")
    print(f"  {len(df)} scanpaths, {df['image_path'].nunique()} unique images")

    # canonical val split — identical to our model and DG3
    train_df, val_df = canonical_split(df)
    print(f"  After val split: {train_df['image_path'].nunique()} train, "
          f"{val_df['image_path'].nunique()} val images")

    # max_images: take first N unique images (same as ScanpathDataset)
    if args.max_images and args.max_images < train_df["image_path"].nunique():
        kept = train_df["image_path"].unique()[:args.max_images]
        train_df = train_df[train_df["image_path"].isin(kept)].reset_index(drop=True)
        print(f"  Subset to {train_df['image_path'].nunique()} train images")

    # cap scanpaths per image
    if args.max_paths_per_image:
        train_df = train_df.groupby("image_path").head(args.max_paths_per_image).reset_index(drop=True)
        val_df = val_df.groupby("image_path").head(args.max_paths_per_image).reset_index(drop=True)

    print(f"  Final: {len(train_df)} train, {len(val_df)} val scanpaths")

    C = args.canvas
    D = args.dummy_duration_ms
    train_entries = to_entries(train_df, "train", C, D)
    val_entries = to_entries(val_df, "validation", C, D)

    # all unique image stems for feature extraction
    all_stems = sorted(set(
        Path(e["name"]).stem for e in train_entries + val_entries
    ))

    ds_name = out.name
    json.dump(train_entries, open(out / f"{ds_name}_fixations_train.json", "w"))
    json.dump(val_entries, open(out / f"{ds_name}_fixations_validation.json", "w"))
    json.dump([], open(out / f"{ds_name}_fixations_test.json", "w"))

    # image list for feature precomputation
    list_path = out / "image_stems.txt"
    with open(list_path, "w") as f:
        for s in all_stems:
            f.write(s + "\n")

    print(f"\nWrote to {out}/:")
    print(f"  train: {len(train_entries)}, val: {len(val_entries)}")
    print(f"  unique images (train+val): {len(all_stems)}")
    print(f"  image list: {list_path}")


if __name__ == "__main__":
    main()
