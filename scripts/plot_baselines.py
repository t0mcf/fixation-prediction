"""plot spatial baseline heatmaps using the same code as eval_baselines.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from scripts.evaluate_baselines import (
    estimate_empirical_density,
    make_center_gaussian_log_heatmap,
    make_uniform_log_heatmap,
)
from src.data.dataloader import make_dataloader


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--parquet-path", default=default_parquet_path)
    parser.add_argument("--imagenet-root", default=default_imagenet_root)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--max-prefix-len", type=int, default=15)

    parser.add_argument("--center-sigma", type=float, default=12.0)
    parser.add_argument("--density-smoothing", type=float, default=1.0)
    parser.add_argument("--density-batches", type=int, default=None)

    parser.add_argument("--output-dir", default="runs/baseline_plots")

    return parser.parse_args()


def save_heatmap(path: Path, heatmap: torch.Tensor, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = heatmap.detach().cpu().float().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(arr, origin="upper")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"saved: {path}", flush=True)


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    print("building uniform baseline...", flush=True)
    uniform_log = make_uniform_log_heatmap(
        size=args.heatmap_size,
        device=device,
    )
    save_heatmap(
        output_dir / "uniform.png",
        uniform_log.exp(),
        "Uniform baseline",
    )

    print("building center gaussian baseline...", flush=True)
    center_log = make_center_gaussian_log_heatmap(
        size=args.heatmap_size,
        sigma=args.center_sigma,
        device=device,
    )
    save_heatmap(
        output_dir / "center_gaussian.png",
        center_log.exp(),
        "Center Gaussian baseline",
    )

    print("building train loader for empirical density...", flush=True)
    train_loader = make_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=args.max_images,
        use_grouped_sampler=False,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    print("building empirical density...", flush=True)
    empirical_log = estimate_empirical_density(
        loader=train_loader,
        heatmap_size=args.heatmap_size,
        sigma=args.density_smoothing,
        device=device,
        max_batches=args.density_batches,
    )

    save_heatmap(
        output_dir / "empirical_density.png",
        empirical_log.exp(),
        "Empirical train density baseline",
    )

    print(f"all baseline plots saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()