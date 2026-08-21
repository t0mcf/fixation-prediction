"""
MIT1003 dataset loader for zero-shot and fine-tuning evaluation.

Returns the same {image, fixations, fixations_len} format as ScanpathDataset
so the existing training / evaluation code works unchanged.

Fixation coordinates are normalised to [-1, 1] relative to the ORIGINAL image
size before any resizing, so that the spatial meaning is preserved when the
image is resized to 224×224 for DINOv2.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD

MIT1003_ROOT = Path("/mnt/vast-nhr/projects/nim00018/datasets/MIT1003")

_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class MIT1003Dataset(Dataset):
    """
    Each sample is one human scanpath on one image.

    Items:
        image         (3, 224, 224)  float32 — ImageNet-normalised
        fixations     (max_seq_len, 2)  float32 — normalised [-1, 1] (x, y)
        fixations_len  int            — number of valid fixations (<= max_seq_len)

    Coordinate convention matches ScanpathDataset:
        x = horizontal, y = vertical, both in [-1, 1].
        (0, 0) = image centre; (-1, -1) = top-left.
    """

    def __init__(
        self,
        root: Path = MIT1003_ROOT,
        max_seq_len: int = 16,
        min_fixations: int = 1,
        image_indices: list[int] | None = None,
        prepend_initial_fixation: bool = True,
    ) -> None:
        super().__init__()
        self.root        = Path(root)
        self.max_seq_len = max_seq_len
        # Prepend a central (0,0) initial fixation to every scanpath. This matches
        # (a) our synthetic pretraining data, where EVERY scanpath starts at (0,0)
        #     center (verified: 100% of synthetic scanpaths, std=0) — so fine-tuning
        #     data must follow the same convention the model was pretrained with;
        # (b) the MIT1003 experimental protocol (subjects fixated a central cross
        #     before each image), as reconstructed by pysaliency's
        #     get_mit1003_with_initial_fixation, which DG3 uses.
        # min_fixations counts REAL fixations; with the prepended centre, a scanpath
        # with 1 real fixation already yields 1 valid prediction step (centre -> fix1).
        self.prepend_initial_fixation = prepend_initial_fixation

        with h5py.File(self.root / "fixations.hdf5", "r") as f:
            xs      = f["train_xs"][:]          # (N, 15) pixel x
            ys      = f["train_ys"][:]          # (N, 15) pixel y
            ns      = f["train_ns"][:]          # (N,)    image index
            subjects = f["train_subjects"][:]   # (N,)

        with h5py.File(self.root / "stimuli.hdf5", "r") as f:
            self._filenames = [
                fn.decode() if isinstance(fn, bytes) else fn
                for fn in f["filenames"][:]
            ]
            shapes = f["shapes"][:]  # (1003, 3) — (H, W, C) per image

        self._img_shapes = {i: (int(shapes[i][0]), int(shapes[i][1]))
                            for i in range(len(shapes))}   # i → (H, W)

        # filter to requested image_indices subset (for train/val splits)
        valid_img = set(image_indices) if image_indices is not None else None

        # build per-scanpath records; drop scanpaths with too few fixations
        records = []
        for i in range(len(xs)):
            img_idx = int(ns[i])
            if valid_img is not None and img_idx not in valid_img:
                continue
            x_row = xs[i]
            y_row = ys[i]
            n_valid = int(np.sum(~np.isnan(x_row)))
            if n_valid < min_fixations:
                continue
            records.append({
                "img_idx":  img_idx,
                "subject":  int(subjects[i]),
                "xs":       x_row,
                "ys":       y_row,
                "n_valid":  n_valid,
            })

        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec     = self._records[idx]
        img_idx = rec["img_idx"]
        H, W    = self._img_shapes[img_idx]

        # load and transform image
        img_path = self.root / self._filenames[img_idx]
        img = Image.open(img_path).convert("RGB")
        image = _img_transform(img)  # (3, 224, 224)

        # offset = 1 if we reserve index 0 for the central initial fixation
        offset = 1 if self.prepend_initial_fixation else 0
        # number of REAL fixations we can fit after the optional centre
        n_real = min(rec["n_valid"], self.max_seq_len - offset)

        # fixations[0] stays (0,0) = centre when prepending (zeros init); real
        # fixations go at indices [offset : offset + n_real].
        fixations = torch.zeros(self.max_seq_len, 2, dtype=torch.float32)

        xs = rec["xs"][:n_real]
        ys = rec["ys"][:n_real]

        # normalise pixel coords to [-1, 1] using ORIGINAL image size
        # x: 0..W-1  ->  -1..+1    y: 0..H-1  ->  -1..+1
        x_norm = np.clip((xs / (W - 1)) * 2.0 - 1.0, -1.0, 1.0)
        y_norm = np.clip((ys / (H - 1)) * 2.0 - 1.0, -1.0, 1.0)

        fixations[offset:offset + n_real, 0] = torch.from_numpy(x_norm.astype(np.float32))
        fixations[offset:offset + n_real, 1] = torch.from_numpy(y_norm.astype(np.float32))

        n_fix = offset + n_real   # total length incl. the centre fixation

        return {
            "image":         image,
            "fixations":     fixations,
            "fixations_len": torch.tensor(n_fix, dtype=torch.long),
            "img_idx":       torch.tensor(img_idx, dtype=torch.long),
        }


def make_mit1003_loader(
    root: Path = MIT1003_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    max_seq_len: int = 16,
    min_fixations: int = 1,
    image_indices: list[int] | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = MIT1003Dataset(
        root=root,
        max_seq_len=max_seq_len,
        min_fixations=min_fixations,
        image_indices=image_indices,
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=g if shuffle else None,
    )
