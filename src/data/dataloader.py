"""DataLoader factory with seed-based worker init."""

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
from .dataset import ScanpathDataset


def _worker_init_fn(worker_id: int):
    """Seed each worker independently for reproducibility."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


class ImageGroupedSampler(Sampler):
    """
    Yields batches of exactly `paths_per_image` scanpaths from the same image.
    Groups the dataset by image, then shuffles groups (not individual samples).
    Drops the last incomplete group for each image.
    """

    def __init__(self, dataset: ScanpathDataset, paths_per_image: int = 16, seed: int = 42):
        self.paths_per_image = paths_per_image
        self.seed = seed
        self.epoch = 0 # track epoch for deterministic shuffling across epochs

        # build image → [indices] map using pandas groupby (vectorised, not a Python loop)
        grouped = dataset.df.groupby("image_path", sort=False).indices
        self.groups = [
            idxs.tolist() for idxs in grouped.values()
            if len(idxs) >= paths_per_image
        ]
        
    def set_epoch(self, epoch: int):
        # call at start of each epoch to shuffle differently across epochs while maintaining reproducibility
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = rng.permutation(len(self.groups))
        for g in group_order:
            idxs = self.groups[g]
            chosen = rng.choice(idxs, size=self.paths_per_image, replace=False)
            yield from chosen.tolist()

    def __len__(self):
        return len(self.groups) * self.paths_per_image


def make_dataloader(
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 8,
    seed: int = 42,
    paths_per_image: int = 16,
    use_grouped_sampler: bool = True,
    max_samples: int = None,
    max_images: int = None,
    prefetch_factor: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    dataset = ScanpathDataset(split=split, seed=seed, max_samples=max_samples,
                              max_images=max_images,
                              heatmap_sigma=dataset_kwargs.pop("heatmap_sigma", 8.0),
                              heatmap_size=dataset_kwargs.pop("heatmap_size", 64),
                              **dataset_kwargs)

    loader_kwargs = dict(
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )

    if use_grouped_sampler and split == "train":
        sampler = ImageGroupedSampler(dataset, paths_per_image=paths_per_image, seed=seed)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          drop_last=True, **loader_kwargs)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      drop_last=(split == "train"), **loader_kwargs)
