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
        self.epoch = 0

        grouped = dataset.df.groupby("image_path", sort=False).indices
        self.groups = [
            idxs.tolist() for idxs in grouped.values()
            if len(idxs) >= paths_per_image
        ]
        if getattr(dataset, "hflip_double", False):
            # mirrored copies live at index + N and are distinct images to the
            # encoder, so they form their own orientation-pure groups
            n = len(dataset.df)
            self.groups += [[i + n for i in g] for g in self.groups]
        # TODO: images with fewer than paths_per_image scanpaths are silently
        # dropped here. This is a workaround for the deduplication assumption in
        # ScanpathModel.forward() — the model expects consecutive blocks of
        # exactly paths_per_image identical images in the batch. The clean fix is
        # to pass image_ids in the batch and use torch.unique(return_inverse=True)
        # inside forward() to deduplicate regardless of batch layout, removing the
        # coupling between sampler and model entirely.

    def set_epoch(self, epoch: int):
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


def image_subset_from_list(list_path: str, parquet_path: str) -> list[str]:
    """Resolve a protocol split file ('class/file.JPEG' per line, e.g.
    thesis_protocol/v2/splits/imagenet_av/test_10k.txt) to the parquet's
    image_path strings, matching on the last two path components exactly as
    train.py does for --train-image-list. Fails loudly on any missing image so
    an evaluation can never silently run on a subset of the split."""
    import pandas as pd

    def _key(p: str) -> str:
        a = p.strip().split("/")
        return a[-2] + "/" + a[-1]

    wanted = {_key(l) for l in open(list_path) if l.strip()}
    available = (pd.read_parquet(parquet_path, engine="pyarrow",
                                 columns=["image_path"])["image_path"]
                 .unique().tolist())
    by_key = {_key(p): p for p in available}
    missing = wanted - by_key.keys()
    if missing:
        raise SystemExit(f"{len(missing)} of {len(wanted)} images from "
                         f"{list_path} are not in {parquet_path}, "
                         f"e.g. {sorted(missing)[:3]}")
    return [by_key[k] for k in sorted(wanted)]


def make_dataloader(
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 8,
    seed: int = 42,
    paths_per_image: int = 16,
    use_grouped_sampler: bool = True,
    max_samples: int = None,
    max_images: int = None,
    max_paths_per_image: int = None,
    max_seq_len: int = 16,
    prefetch_factor: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    dataset = ScanpathDataset(
        split=split,
        seed=seed,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        max_images=max_images,
        max_paths_per_image=max_paths_per_image,
        **dataset_kwargs,
    )

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
