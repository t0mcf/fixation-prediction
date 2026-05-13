"""evaluate simple spatial baselines for scanpath prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.training.losses import kl_loss, ll_score
from src.training.metrics import nss_score, auc_score


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def make_uniform_log_pred(batch_size: int, heatmap_size: int, device: str) -> torch.Tensor:
    """
    Uniform spatial baseline.

    Predicts the same probability for every heatmap location.
    """
    H = W = heatmap_size
    logits = torch.zeros(batch_size, H, W, device=device)
    return F.log_softmax(logits.view(batch_size, -1), dim=-1).view(batch_size, H, W)


def make_center_gaussian_log_pred(
    batch_size: int,
    heatmap_size: int,
    sigma: float,
    device: str,
) -> torch.Tensor:
    """
    Center Gaussian baseline.

    Predicts a fixed Gaussian distribution centered in the image.
    """
    H = W = heatmap_size

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0

    logits = -((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2)

    logits = logits.unsqueeze(0).expand(batch_size, H, W)

    return F.log_softmax(logits.view(batch_size, -1), dim=-1).view(batch_size, H, W)


def update_density_counts(
    counts: torch.Tensor,
    target_xy: torch.Tensor,
) -> None:
    """
    Add target fixation locations to a spatial count map.

    counts:
        (H, W)

    target_xy:
        (B, 2), normalized coordinates in [-1, 1]
    """
    H, W = counts.shape
    device = counts.device

    x = target_xy[:, 0].to(device)
    y = target_xy[:, 1].to(device)

    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    for r, c in zip(row, col):
        counts[r, c] += 1.0


def build_empirical_density(
    loader: torch.utils.data.DataLoader,
    heatmap_size: int,
    device: str,
    max_batches: int | None = None,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """
    Build empirical fixation density from training targets only.

    Returns:
        log_density: (H, W)
    """
    counts = torch.full(
        (heatmap_size, heatmap_size),
        fill_value=smoothing,
        dtype=torch.float32,
        device=device,
    )

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        target_xy = batch["target_xy"].to(device)
        update_density_counts(counts, target_xy)

    probs = counts / counts.sum()
    log_density = torch.log(probs.clamp_min(1e-8))

    return log_density


def evaluate_log_pred_baseline(
    loader: torch.utils.data.DataLoader,
    name: str,
    make_log_pred,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            heatmap = batch["heatmap"].to(device)
            target_xy = batch["target_xy"].to(device)
            batch_size = heatmap.shape[0]

            log_pred = make_log_pred(batch_size)

            value_kl = kl_loss(log_pred, heatmap)
            value_ll = ll_score(log_pred, target_xy)
            value_nss = nss_score(log_pred, target_xy)
            value_auc = auc_score(log_pred, target_xy)

            total_kl += value_kl.item() * batch_size
            total_ll += value_ll.item() * batch_size
            total_nss += value_nss.item() * batch_size
            total_auc += value_auc.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {
            "name": name,
            "samples": 0,
            "kl": float("nan"),
            "ll": float("nan"),
            "nss": float("nan"),
            "auc": float("nan"),
        }

    metrics = {
        "name": name,
        "samples": total_samples,
        "kl": total_kl / total_samples,
        "ll": total_ll / total_samples,
        "nss": total_nss / total_samples,
        "auc": total_auc / total_samples,
    }

    return metrics


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
    parser.add_argument("--eval-batches", type=int, default=50)

    return parser.parse_args()


def print_metrics(metrics: dict[str, float]) -> None:
    print(
        f"{metrics['name']:>18} | "
        f"samples={int(metrics['samples']):5d} | "
        f"kl={metrics['kl']:.4f} | "
        f"ll={metrics['ll']:.4f} | "
        f"nss={metrics['nss']:.4f} | "
        f"auc={metrics['auc']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    train_loader = make_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=args.max_images,
        use_grouped_sampler=False,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        max_prefix_len=args.max_prefix_len,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=args.max_images,
        use_grouped_sampler=False,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        max_prefix_len=args.max_prefix_len,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    print("building empirical train density...", flush=True)
    train_log_density = build_empirical_density(
        loader=train_loader,
        heatmap_size=args.heatmap_size,
        device=device,
        max_batches=args.density_batches,
        smoothing=args.density_smoothing,
    )

    print("evaluating baselines...", flush=True)

    baselines = []

    baselines.append(
        evaluate_log_pred_baseline(
            loader=val_loader,
            name="uniform",
            make_log_pred=lambda B: make_uniform_log_pred(
                batch_size=B,
                heatmap_size=args.heatmap_size,
                device=device,
            ),
            device=device,
            max_batches=args.eval_batches,
        )
    )

    baselines.append(
        evaluate_log_pred_baseline(
            loader=val_loader,
            name="center_gaussian",
            make_log_pred=lambda B: make_center_gaussian_log_pred(
                batch_size=B,
                heatmap_size=args.heatmap_size,
                sigma=args.center_sigma,
                device=device,
            ),
            device=device,
            max_batches=args.eval_batches,
        )
    )

    baselines.append(
        evaluate_log_pred_baseline(
            loader=val_loader,
            name="empirical_density",
            make_log_pred=lambda B: train_log_density.unsqueeze(0).expand(
                B,
                args.heatmap_size,
                args.heatmap_size,
            ),
            device=device,
            max_batches=args.eval_batches,
        )
    )

    print("\nsummary", flush=True)
    for metrics in baselines:
        print_metrics(metrics)


if __name__ == "__main__":
    main()