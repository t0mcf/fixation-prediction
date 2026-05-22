"""Scanpath dataset: parquet → (image, fixations, fixations_len)."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import functools


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


_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ScanpathDataset(Dataset):
    """
    Each sample is one full scanpath: (image, fixations, fixations_len).

    Returns the complete fixation sequence padded to max_seq_len.
    The training loop is responsible for constructing prefix/target pairs
    and computing target heatmaps.

    Args:
        split: "train" or "val"
        seed: random seed for 90/10 image split
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
    ):
        assert split in ("train", "val")
        self.split = split
        self.max_seq_len = max_seq_len
        self.imagenet_root = imagenet_root

        df = pd.read_parquet(parquet_path, engine="pyarrow")

        # deterministic 90/10 split by image
        rng = np.random.default_rng(seed)
        images = np.array(df["image_path"].unique())
        rng.shuffle(images)
        n_val = max(1, int(len(images) * 0.1))
        val_images = set(images[:n_val])

        if split == "train":
            self.df = df[~df["image_path"].isin(val_images)].reset_index(drop=True)
        else:
            self.df = df[df["image_path"].isin(val_images)].reset_index(drop=True)

        if max_images:
            kept = self.df["image_path"].unique()[:max_images]
            self.df = self.df[self.df["image_path"].isin(kept)].reset_index(drop=True)
        elif max_samples and len(self.df) > max_samples:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

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
