"""Scanpath dataset: parquet → (image, fixations, fixations_len)."""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import functools

from .constants import IMAGENET_MEAN, IMAGENET_STD


IMAGENET_ROOT = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"
PARQUET_PATH = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

HEATMAP_SIZE = 64


@functools.lru_cache(maxsize=32)
def _load_image_cached(path: str) -> torch.Tensor:
    """Load, resize, and normalise an image. Cache per worker process (LRU, 32 images)."""
    img = Image.open(path).convert("RGB")
    return _img_transform(img)


# Default: direct resize (what the whole ladder was trained with). Set
# OURS_CROP_TRANSFORM=1 to use Resize(256)+CenterCrop(224) instead — the
# standard ImageNet transform that DG3 and the AV generator use. Used to test
# whether the ours-vs-DG3 coordinate-frame mismatch (engineering_log §19.17)
# materially changes ours' numbers. Does NOT change default behaviour.
if os.environ.get("OURS_CROP_TRANSFORM") == "1":
    _img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
else:
    _img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ScanpathDataset(Dataset):
    """
    Each sample is one full scanpath: (image, fixations, fixations_len).

    Returns the complete fixation sequence padded to max_seq_len.
    The training loop is responsible for constructing prefix/target pairs
    and computing target heatmaps.

    Args:
        split: "train" or "val"
        seed: legacy parameter; the val split is now fixed (5000 images by
              seed=3141, matching DG3). still accepted for backward compat and
              used downstream by the dataloader/sampler.
        max_seq_len: maximum total scanpath length (pad/truncate to this)
        imagenet_root: override for ImageNet root path
        parquet_path: override for parquet path
        max_samples: limit total number of rows
        max_images: limit by unique image count
        max_paths_per_image: cap scanpaths per image (deterministic: keeps first N rows
                             per image after sorting by image_path)
    """

    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        max_seq_len: int = 16,
        imagenet_root: str = IMAGENET_ROOT,
        parquet_path: str = PARQUET_PATH,
        max_samples: int = None,
        max_images: int = None,
        max_paths_per_image: int = None,
        image_subset=None,
        epoch_subset=None,
    ):
        """
        Canonical val split: 5000 images selected by seed=3141, matching DG3's
        scanpather/parquet_to_pysaliency.py exactly. Both DG3 and our models hold
        these same images out from training so training val metrics are directly
        comparable, and the same set serves as the canonical test set for
        evaluate_checkpoint.py. The `seed` argument no longer controls the val
        split (which is fixed by design); it remains in the signature for
        backward compatibility and is still used by the dataloader/sampler.

        NOTE: runs trained before this change used a seed=42, 90/10 split. Those
        checkpoints' "val" metrics during training were on a different set and
        their training data overlaps the canonical val set (severely for the
        800k model; ~90% of canonical val falls in any seed=42 training pool).

        image_subset: explicit iterable of image_path values. When given, the
                      internal canonical split (and max_images/max_samples) are
                      bypassed entirely and exactly these images are used.
        epoch_subset: explicit iterable of epoch (subject) IDs to keep. Applied
                      after image selection, matching DG3 _cap_scanpaths.
        """
        assert split in ("train", "val")
        self.split = split
        self.max_seq_len = max_seq_len
        self.imagenet_root = imagenet_root

        df = pd.read_parquet(parquet_path, engine="pyarrow")

        if image_subset is not None:
            # explicit image list — bypass the internal canonical split entirely
            subset = set(image_subset)
            self.df = df[df["image_path"].isin(subset)].reset_index(drop=True)
        else:
            # Canonical val split, identical to DG3's selection:
            # key = last two path components ("class/image.JPEG"), which gives
            # the same sort order as DG3's transformed absolute paths (their
            # IMAGE_BASE prefix is constant and drops out of the sort). Verified
            # to produce the exact same 5000-image val set as DG3 (seed=3141).
            def _canonical_key(p):
                parts = p.split('/')
                return parts[-2] + '/' + parts[-1]

            keys = df["image_path"].map(_canonical_key)
            unique_keys = sorted(keys.unique())
            rng_canon = np.random.default_rng(3141)
            val_keys = set(
                unique_keys[i]
                for i in rng_canon.permutation(len(unique_keys))[:5000]
            )
            is_val = keys.isin(val_keys)

            if split == "train":
                self.df = df[~is_val].reset_index(drop=True)
            else:
                self.df = df[is_val].reset_index(drop=True)

            if max_images:
                kept = self.df["image_path"].unique()[:max_images]
                self.df = self.df[self.df["image_path"].isin(kept)].reset_index(drop=True)
            elif max_samples and len(self.df) > max_samples:
                self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # epoch_subset: keep only specified scanpath/subject IDs.
        # matches DG3 _cap_scanpaths (global epoch filter, applied after image split).
        if epoch_subset is not None:
            self.df = self.df[
                self.df["epoch"].isin(list(epoch_subset))
            ].reset_index(drop=True)

        if max_paths_per_image:
            self.df = (
                self.df.groupby("image_path", sort=False)
                .head(max_paths_per_image)
                .reset_index(drop=True)
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fixations = row["locations"].reshape(-1, 2)  # (N, 2)

        # truncate to max_seq_len
        fixations_len = min(len(fixations), self.max_seq_len)

        # pad to max_seq_len with zeros
        fixation_tensor = torch.zeros(self.max_seq_len, 2, dtype=torch.float32)
        fixation_tensor[:fixations_len] = torch.tensor(
            fixations[:fixations_len], dtype=torch.float32
        )

        rel_path = row["image_path"].replace("./data/ImageNet", self.imagenet_root)
        img_tensor = _load_image_cached(rel_path)

        return {
            "image": img_tensor,           # (3, 224, 224)
            "fixations": fixation_tensor,  # (max_seq_len, 2)
            "fixations_len": fixations_len, # int
        }
