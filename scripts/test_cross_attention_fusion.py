"""test cross-attention fusion between image tokens and scanpath tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import CrossAttentionFusion, ScanpathEncoder


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

    d_model = 256
    n_patches = 256

    scanpath_encoder = ScanpathEncoder(
        d_model=d_model,
        n_heads=8,
        n_layers=2,
        max_prefix_len=15,
    ).to(device)

    fusion = CrossAttentionFusion(
        d_model=d_model,
        n_heads=8,
        dropout=0.1,
    ).to(device)

    scanpath_tokens = scanpath_encoder(prefix, prefix_len)

    # fake image tokens for this isolated fusion test: (B, N, d_model)
    image_tokens = torch.randn(
        prefix.shape[0],
        n_patches,
        d_model,
        device=device,
    )

    fused_tokens = fusion(
        image_tokens=image_tokens,
        scanpath_tokens=scanpath_tokens,
        prefix_len=prefix_len,
    )

    print("image tokens shape:", tuple(image_tokens.shape), flush=True)
    print("scanpath tokens shape:", tuple(scanpath_tokens.shape), flush=True)
    print("fused tokens shape:", tuple(fused_tokens.shape), flush=True)

    expected_shape = image_tokens.shape
    assert tuple(fused_tokens.shape) == tuple(expected_shape), (
        f"expected {tuple(expected_shape)}, got {tuple(fused_tokens.shape)}"
    )

    assert torch.isfinite(fused_tokens).all(), "fused tokens contain NaN or Inf"

    print(
        "fused tokens mean/std:",
        float(fused_tokens.detach().mean()),
        float(fused_tokens.detach().std()),
        flush=True,
    )

    print("cross-attention fusion test passed", flush=True)


if __name__ == "__main__":
    main()