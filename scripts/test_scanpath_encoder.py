"""test the scanpath encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathEncoder


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    loader = make_dataloader(
        split="train",
        batch_size=4,
        num_workers=0,
        max_images=5,
        use_grouped_sampler=False,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )

    batch = next(iter(loader))
    prefix = batch["prefix"].to(device)
    prefix_len = batch["prefix_len"].to(device)

    # prefix: (B, T, 2), prefix_len: (B,)
    print("prefix shape:", tuple(prefix.shape), flush=True)
    print("prefix_len:", prefix_len.tolist(), flush=True)

    encoder = ScanpathEncoder(
        d_model=256,
        n_heads=8,
        n_layers=2,
        max_prefix_len=15,
    ).to(device)

    tokens = encoder(prefix, prefix_len)

    # tokens: (B, T, d_model)
    print("tokens shape:", tuple(tokens.shape), flush=True)
    print("tokens mean/std:", float(tokens.mean()), float(tokens.std()), flush=True)

    expected_shape = (prefix.shape[0], prefix.shape[1], 256)
    assert tuple(tokens.shape) == expected_shape, (
        f"expected {expected_shape}, got {tuple(tokens.shape)}"
    )

    assert torch.isfinite(tokens).all(), "scanpath tokens contain NaN or Inf"

    print("scanpath encoder test passed", flush=True)


if __name__ == "__main__":
    main()