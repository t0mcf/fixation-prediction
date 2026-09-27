"""
parquet_to_pysaliency.py
========================
Converts Parquet scanpaths to PySaliency objects for DeepGaze training.
"""
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import pysaliency
from tqdm import tqdm


def load_parquet(
    parquet_path: Path,
    image_base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise RuntimeError(f"Parquet file is empty: {parquet_path}")
    if image_base_dir is not None:
        df = df.copy()
        paths = df["image_path"].str.split('/')
        df["image_path"] = str(image_base_dir) + "/" + paths.str[-2] + "/" + paths.str[-1]
    return df


def _cap_scanpaths(df: pd.DataFrame, n: int, param_name: str) -> pd.DataFrame:
    """Keep only the first *n* epoch IDs in *df*. Warns if fewer than *n* exist."""
    available = sorted(df["epoch"].unique())
    if len(available) < n:
        warnings.warn(
            f"{param_name}={n} requested but only {len(available)} subjects exist "
            "in the parquet.",
            UserWarning, stacklevel=3,
        )
    return df[df["epoch"].isin(available[:n])]


def split_images(
    df: pd.DataFrame,
    train_frac: float,
    seed: int,
    train_n_images: Optional[int] = None,
    val_n_images: Optional[int] = None,
    train_n_scanpaths: Optional[int] = None,
    val_n_scanpaths: Optional[int] = None,
):
    """Split image paths into train and validation sets.

    When *val_n_images* is given the validation set is chosen first (fixed by
    *seed* alone, independent of *train_n_images*), so the same val images are
    used regardless of how many training images are selected.  The training
    pool is the complement of the val set, optionally further limited by
    *train_n_images*.

    When *val_n_images* is None the legacy behaviour applies: train/val are
    split from the (optionally subsetted) image pool by *train_frac*.

    *train_n_scanpaths* and *val_n_scanpaths* independently cap how many
    scanpaths (subjects / epochs) are kept per image on each split.  Both are
    applied after the image split, so they are fully independent of each other
    and of how many subjects exist in the raw data.

    Returns
    -------
    train_df, train_imgs, val_df, val_imgs
    """
    rng = np.random.default_rng(seed)
    all_image_paths = sorted(df["image_path"].unique())

    if val_n_images is not None:
        # --- Fixed val set (independent of train_n_images) ---
        n_val = min(val_n_images, len(all_image_paths))
        val_imgs = [all_image_paths[i]
                    for i in rng.permutation(len(all_image_paths))[:n_val]]
        val_set = set(val_imgs)

        train_pool = [p for p in all_image_paths if p not in val_set]
        if train_n_images is not None and train_n_images < len(train_pool):
            train_imgs = [train_pool[i]
                          for i in rng.permutation(len(train_pool))[:train_n_images]]
        else:
            train_imgs = train_pool
    else:
        # --- Legacy: random-fraction split (val depends on train_n_images) ---
        image_paths = all_image_paths
        if train_n_images is not None and train_n_images < len(image_paths):
            image_paths = [image_paths[i]
                           for i in rng.permutation(len(image_paths))[:train_n_images]]
        idx = rng.permutation(len(image_paths))
        n_train = max(1, int(len(image_paths) * train_frac))
        train_imgs = [image_paths[i] for i in idx[:n_train]]
        val_imgs   = [image_paths[i] for i in idx[n_train:]] or train_imgs

    train_df = df[df["image_path"].isin(set(train_imgs))]
    val_df   = df[df["image_path"].isin(set(val_imgs))]

    if train_n_scanpaths is not None:
        train_df = _cap_scanpaths(train_df, train_n_scanpaths, "train_n_scanpaths")
    if val_n_scanpaths is not None:
        val_df = _cap_scanpaths(val_df, val_n_scanpaths, "val_n_scanpaths")

    return train_df, train_imgs, val_df, val_imgs


def build_pysaliency(df: pd.DataFrame, img_list: list, target_size: int = 224):
    img_index = {p: i for i, p in enumerate(img_list)}
    subset = df[df["image_path"].isin(img_index)]
    train_xs, train_ys, train_ts, train_ns, train_subjects = [], [], [], [], []

    for row in tqdm(subset.to_dict('records'), desc="building fixations"):
        n = img_index[row["image_path"]]
        locs = np.array(row["locations"], dtype=np.float32).reshape(-1, 2)
        xs = ((locs[:, 0] + 1.0) * 0.5 * target_size).clip(0, target_size - 1)
        ys = ((locs[:, 1] + 1.0) * 0.5 * target_size).clip(0, target_size - 1)
        ts = np.arange(len(xs), dtype=np.float32)
        train_xs.append(xs)
        train_ys.append(ys)
        train_ts.append(ts)
        train_ns.append(n)
        train_subjects.append(int(row["epoch"]))

    shapes = [(target_size, target_size, 3) for _ in img_list]
    stimuli = pysaliency.FileStimuli(filenames=img_list, shapes=shapes)
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        xs=train_xs, ys=train_ys, ts=train_ts, ns=train_ns, subjects=train_subjects,
    )
    # Drop the 0th fixation of every scanpath (always the forced center start).
    # It is still used as conditioning context in x_hist for subsequent fixations,
    # but training to *predict* it provides no signal and causes all-NaN history
    # which crashes FlexibleScanpathHistoryEncoding.
    fixations = fixations[fixations.lengths > 0]
    return stimuli, fixations


def parquet_to_pysaliency(
    parquet_path: Path,
    train_frac: float = 0.9,
    seed: int = 3141,
    image_base_dir: Optional[Path] = None,
    target_size: int = 224,
    train_n_images: Optional[int] = None,
    val_n_images: Optional[int] = None,
    train_n_scanpaths: Optional[int] = None,
    val_n_scanpaths: Optional[int] = None,
    val_image_list: Optional[Path] = None,
):
    df = load_parquet(Path(parquet_path), image_base_dir)
    if val_image_list is not None:
        # Explicit evaluation list (protocol split file, 'class/file.JPEG' per
        # line, e.g. v2 test_10k.txt): bypasses the seed-based selection.
        # Match on the last two path components; fail loudly on anything
        # missing so an evaluation never silently runs on a subset.
        def _key(p: str) -> str:
            a = str(p).strip().split("/")
            return a[-2] + "/" + a[-1]
        wanted = {_key(l) for l in open(val_image_list) if l.strip()}
        by_key = {_key(p): p for p in df["image_path"].unique()}
        missing = wanted - by_key.keys()
        if missing:
            raise SystemExit(f"{len(missing)} of {len(wanted)} images from "
                             f"{val_image_list} not in {parquet_path}, "
                             f"e.g. {sorted(missing)[:3]}")
        val_imgs = [by_key[k] for k in sorted(wanted)]
        val_df = df[df["image_path"].isin(set(val_imgs))]
        if val_n_scanpaths is not None:
            val_df = _cap_scanpaths(val_df, val_n_scanpaths, "val_n_scanpaths")
        # evaluation-only mode: a one-image placeholder train side
        train_imgs = val_imgs[:1]
        train_df = df[df["image_path"].isin(set(train_imgs))]
        print(f"[parquet_to_pysaliency] explicit val list: {len(val_imgs)} images "
              f"from {val_image_list} (seed-based split bypassed)", flush=True)
    else:
        train_df, train_imgs, val_df, val_imgs = split_images(
            df, train_frac, seed,
            train_n_images=train_n_images,
            val_n_images=val_n_images,
            train_n_scanpaths=train_n_scanpaths,
            val_n_scanpaths=val_n_scanpaths,
        )
    train_stimuli, train_fixations = build_pysaliency(train_df, train_imgs, target_size=target_size)
    val_stimuli,   val_fixations   = build_pysaliency(val_df,   val_imgs,   target_size=target_size)
    return train_stimuli, train_fixations, val_stimuli, val_fixations
