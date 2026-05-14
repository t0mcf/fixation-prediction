"""Scanpath dataset: parquet → (image, fixation_prefix, next_fixation_heatmap)."""

import functools
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_ROOT = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"
PARQUET_PATH = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

HEATMAP_SIZE = 64
GAUSSIAN_SIGMA = 8.0  # pixels in 64×64 space (~12% of image width — encodes "region" not "exact pixel")


def _make_gaussian_heatmap(x: float, y: float, size: int = HEATMAP_SIZE, sigma: float = GAUSSIAN_SIGMA) -> torch.Tensor:
    """x, y in [-1, 1] normalised coords → (size, size) probability heatmap."""
    # convert to pixel coords
    px = (x + 1.0) / 2.0 * (size - 1)
    py = (y + 1.0) / 2.0 * (size - 1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    heatmap = torch.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / heatmap.sum()
    return heatmap


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
    Each sample is one (image, prefix, target) triple drawn from a scanpath.

    For a scanpath of length N, we randomly pick a step t in [1, N-1] and
    return fixations[0:t] as prefix and fixations[t] as the target heatmap.
    Args:
        split: "train" or "val"
        seed: random seed for 90/10 image split 
        max_prefix_len: maximum number of prefix fixations (pad/truncate to this)
        imagenet_root: override for ImageNet root path
        parquet_path: override for parquet path
    """

    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        max_prefix_len: int = 15,
        imagenet_root: str = IMAGENET_ROOT,
        parquet_path: str = PARQUET_PATH,
        max_samples: int = None,
        max_images: int = None,
        heatmap_sigma: float = GAUSSIAN_SIGMA,
        heatmap_size: int = HEATMAP_SIZE,
    ):
        assert split in ("train", "val")
        self.split = split
        self.max_prefix_len = max_prefix_len
        self.imagenet_root = imagenet_root
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size

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

        # max_images: limit by unique image count 
        if max_images:
            kept = self.df["image_path"].unique()[:max_images]
            self.df = self.df[self.df["image_path"].isin(kept)].reset_index(drop=True)
        elif max_samples and len(self.df) > max_samples:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        self._rng = np.random.default_rng(seed + (0 if split == "train" else 1))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        locs = row["locations"]  # flat numpy array length 32 → 16 (x,y) pairs

        fixations = locs.reshape(-1, 2)  # (16, 2)
        n = len(fixations)

        # pick a random step in [1, n-1]
        t = int(self._rng.integers(1, n))

        prefix = fixations[:t]   # (t, 2)
        target_xy = fixations[t]  # (2,)

        # load image — cached per worker process (16 paths share same image → ~16× fewer reads)
        rel_path = row["image_path"].replace("./data/ImageNet", self.imagenet_root)
        img_tensor = _load_image_cached(rel_path)

        # encode prefix as (max_prefix_len, 2) padded tensor + length
        prefix_tensor = torch.zeros(self.max_prefix_len, 2, dtype=torch.float32)
        prefix_len = min(t, self.max_prefix_len)
        prefix_tensor[:prefix_len] = torch.tensor(prefix[-prefix_len:], dtype=torch.float32)

        # target heatmap
        heatmap = _make_gaussian_heatmap(float(target_xy[0]), float(target_xy[1]),
                                         size=self.heatmap_size, sigma=self.heatmap_sigma)

        return {
            "image": img_tensor,           # (3, 224, 224)
            "prefix": prefix_tensor,       # (max_prefix_len, 2)
            "prefix_len": prefix_len,      # int
            "heatmap": heatmap,            # (64, 64)
            "target_xy": torch.tensor(target_xy, dtype=torch.float32),  # (2,) in [-1,1]
        }

    def reseed_rng(self, worker_id: int) -> None:   #to be called from DataLoader worker_init_fn to ensure different random steps across workers while maintaining reproducibility
        self._rng = np.random.default_rng(
            [torch.initial_seed() % (2 ** 32), worker_id]
        )
