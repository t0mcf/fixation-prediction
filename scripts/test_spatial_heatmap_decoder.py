"""test the spatial heatmap decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.models.dino_scanpather import SpatialHeatmapDecoder


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    B = 4
    N = 256
    d_model = 256

    decoder = SpatialHeatmapDecoder(
        d_model=d_model,
        patch_grid_size=16,
        heatmap_size=64,
        hidden_channels=256,
    ).to(device)

    patch_tokens = torch.randn(B, N, d_model, device=device)

    logits = decoder(patch_tokens)

    # logits: (B, 64, 64)
    print("patch tokens shape:", tuple(patch_tokens.shape), flush=True)
    print("logits shape:", tuple(logits.shape), flush=True)
    print(
        "logits mean/std:",
        float(logits.detach().mean()),
        float(logits.detach().std()),
        flush=True,
    )

    expected_shape = (B, 64, 64)
    assert tuple(logits.shape) == expected_shape, (
        f"expected {expected_shape}, got {tuple(logits.shape)}"
    )

    assert torch.isfinite(logits).all(), "logits contain NaN or Inf"

    print("spatial heatmap decoder test passed", flush=True)


if __name__ == "__main__":
    main()