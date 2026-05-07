"""test the full scanpath model forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    loader = make_dataloader(
        split="train",
        batch_size=1,
        num_workers=0,
        max_images=5,
        use_grouped_sampler=False,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )

    batch = next(iter(loader))

    image = batch["image"].to(device)
    prefix = batch["prefix"].to(device)
    prefix_len = batch["prefix_len"].to(device)

    print("image shape:", tuple(image.shape), flush=True)
    print("prefix shape:", tuple(prefix.shape), flush=True)
    print("prefix_len:", prefix_len.tolist(), flush=True)

    model = ScanpathModel(
        visual_encoder="dinov2_base",
        pretrained_encoder=True,
        img_size=224,
        d_model=256,
        n_heads=8,
        scanpath_layers=2,
        max_prefix_len=15,
        heatmap_size=64,
        dropout=0.1,
        decoder_hidden_channels=256,
    ).to(device)

    model.eval()

    total_params, trainable_params = count_params(model)
    print("total params:", total_params, flush=True)
    print("trainable params:", trainable_params, flush=True)

    with torch.no_grad():
        log_heatmap = model(image, prefix, prefix_len)

    # log_heatmap: (B, 64, 64)
    print("log_heatmap shape:", tuple(log_heatmap.shape), flush=True)
    print(
        "log_heatmap mean/std:",
        float(log_heatmap.mean()),
        float(log_heatmap.std()),
        flush=True,
    )

    expected_shape = (image.shape[0], 64, 64)
    assert tuple(log_heatmap.shape) == expected_shape, (
        f"expected {expected_shape}, got {tuple(log_heatmap.shape)}"
    )

    assert torch.isfinite(log_heatmap).all(), "log_heatmap contains NaN or Inf"

    probs = log_heatmap.exp()

    # probs: (B, 64, 64), each sample should sum to 1 over spatial locations
    probs_sum = probs.sum(dim=(-1, -2))
    print("probability sums:", probs_sum, flush=True)

    assert torch.allclose(
        probs_sum,
        torch.ones_like(probs_sum),
        atol=1e-5,
    ), "probability heatmaps do not sum to 1"

    print("full model forward test passed", flush=True)


if __name__ == "__main__":
    main()