"""Evaluate spatial baselines on the validation set.

Baselines (all image-independent — same heatmap for every image and step):

  uniform          log(1 / (H*W)) everywhere. chance-level predictor.
                   NSS = 0, AUC = 0.5 by construction.

  center_gaussian  fixed isotropic Gaussian centred at the image centre.
                   captures the well-known centre bias in natural image viewing
                   without any learned parameters.

  empirical        average fixation density over a sample of the training set,
                   smoothed with a small Gaussian. captures centre bias plus
                   any global tendencies (e.g. faces tend to appear near centre,
                   text near the bottom, etc.). this is the natural upper bound
                   for a purely spatial, image-independent predictor.

Usage:
    python -m scripts.evaluate_baselines \\
        --parquet-path <path> --imagenet-root <path> \\
        [--max-images 5000] [--batch-size 32] [--eval-batches 200]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.training.losses import make_gaussian_heatmaps_batch
from src.training.metrics import nss_score, auc_score
from src.training.losses import kl_loss, ll_score


DEFAULT_PARQUET = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)
DEFAULT_IMAGENET = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-path", default=DEFAULT_PARQUET)
    p.add_argument("--imagenet-root", default=DEFAULT_IMAGENET)
    p.add_argument("--max-images", type=int, default=5000,
                   help="images used for both density estimation and evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-batches", type=int, default=200,
                   help="val batches to evaluate; -1 = full val set")
    p.add_argument("--density-batches", type=int, default=500,
                   help="train batches used to estimate empirical density")
    p.add_argument("--heatmap-size", type=int, default=64)
    p.add_argument("--heatmap-sigma", type=float, default=2.0,
                   help="sigma for KL target heatmaps and empirical density smoothing")
    p.add_argument("--center-sigma", type=float, default=16.0,
                   help="sigma (in heatmap pixels) for the centre gaussian baseline")
    p.add_argument("--max-seq-len", type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Baseline heatmap builders
# ---------------------------------------------------------------------------

def make_uniform_log_heatmap(size: int, device: str) -> torch.Tensor:
    """Returns a (H, W) log-probability heatmap with uniform distribution."""
    log_p = torch.full((size, size), -math.log(size * size), device=device)
    return log_p


def make_center_gaussian_log_heatmap(size: int, sigma: float, device: str) -> torch.Tensor:
    """Returns a (H, W) log-probability heatmap with a Gaussian centred at (size/2, size/2)."""
    grid = torch.arange(size, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    cx = cy = (size - 1) / 2.0
    heatmap = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / heatmap.sum()
    return torch.log(heatmap.clamp_min(1e-10))


def estimate_empirical_density(
    loader: torch.utils.data.DataLoader,
    heatmap_size: int,
    sigma: float,
    device: str,
    max_batches: int,
) -> torch.Tensor:
    """
    Accumulate all fixation locations from a sample of the dataset into a 2D
    histogram, then smooth with a Gaussian. Returns a (H, W) log-probability map.

    This is an image-independent spatial prior: the average location where
    fixations fall across all images and observers.
    """
    density = torch.zeros(heatmap_size, heatmap_size, device=device)

    print(f"  estimating empirical density from up to {max_batches} batches...")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        fixations = batch["fixations"].to(device)      # (B, T, 2)
        fixations_len = batch["fixations_len"].to(device)  # (B,)

        B, T, _ = fixations.shape

        # vectorised histogram accumulation
        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < fixations_len.unsqueeze(1)
        )  # (B, T)

        x = fixations[..., 0]  # (B, T)
        y = fixations[..., 1]
        col = ((x + 1.0) / 2.0 * (heatmap_size - 1)).round().long().clamp(0, heatmap_size - 1)
        row = ((y + 1.0) / 2.0 * (heatmap_size - 1)).round().long().clamp(0, heatmap_size - 1)

        # flatten and mask
        col_flat = col[valid_mask]
        row_flat = row[valid_mask]
        idx_flat = row_flat * heatmap_size + col_flat

        density.view(-1).scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.float32))

        if (batch_idx + 1) % 100 == 0:
            print(f"    density batches: {batch_idx + 1}/{max_batches}", flush=True)

    if density.sum() == 0:
        print("  WARNING: no fixations found for empirical density, falling back to uniform")
        return make_uniform_log_heatmap(heatmap_size, device)

    # smooth with Gaussian kernel (same sigma as training targets for consistency)
    k_size = int(6 * sigma) | 1  # odd kernel size, at least 6*sigma wide
    k_size = max(k_size, 3)
    grid = torch.arange(k_size, dtype=torch.float32, device=device) - k_size // 2
    kernel_1d = torch.exp(-grid ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (k, k)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)              # (1, 1, k, k)

    density_4d = density.unsqueeze(0).unsqueeze(0)               # (1, 1, H, W)
    density_smooth = F.conv2d(density_4d, kernel_2d, padding=k_size // 2).squeeze()
    density_smooth = density_smooth.clamp_min(0)
    density_smooth = density_smooth / density_smooth.sum()

    print(f"  empirical density estimated (peak at pixel "
          f"{density_smooth.argmax().item() // heatmap_size}, "
          f"{density_smooth.argmax().item() % heatmap_size})", flush=True)

    return torch.log(density_smooth.clamp_min(1e-10))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(
    log_heatmap: torch.Tensor,  # (H, W) — the static baseline prediction
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    heatmap_size: int,
    heatmap_sigma: float,
) -> dict[str, float]:
    """Evaluate a static (image-independent) baseline log-heatmap on the val set."""
    total_kl = total_ll = total_nss = total_auc = 0.0
    total_valid = 0

    log_heatmap = log_heatmap.to(device)

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        fixations = batch["fixations"].to(device)        # (B, T, 2)
        fixations_len = batch["fixations_len"].to(device)  # (B,)

        B, max_seq_len, _ = fixations.shape
        T = max_seq_len - 1
        H = W = heatmap_size

        # expand static heatmap to (B, T, H, W) — same prediction for every sample/step
        log_preds = log_heatmap.unsqueeze(0).unsqueeze(0).expand(B, T, H, W)

        target_fixations = fixations[:, 1:, :]  # (B, T, 2)
        target_heatmaps = make_gaussian_heatmaps_batch(
            target_fixations, heatmap_size, heatmap_sigma
        )

        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < (fixations_len - 1).unsqueeze(1)
        )
        valid_flat = valid_mask.view(-1)

        if not valid_flat.any():
            continue

        lp  = log_preds.reshape(B * T, H, W)[valid_flat]
        th  = target_heatmaps.reshape(B * T, H, W)[valid_flat]
        txy = target_fixations.reshape(B * T, 2)[valid_flat]
        n   = valid_flat.sum().item()

        total_kl  += kl_loss(lp, th).item()  * n
        total_ll  += ll_score(lp, txy).item() * n
        total_nss += nss_score(lp, txy).item() * n
        total_auc += auc_score(lp, txy).item() * n
        total_valid += n

    if total_valid == 0:
        return {"steps": 0, "kl": float("nan"), "ll": float("nan"),
                "nss": float("nan"), "auc": float("nan")}

    return {
        "steps": total_valid,
        "kl":  total_kl  / total_valid,
        "ll":  total_ll  / total_valid,
        "nss": total_nss / total_valid,
        "auc": total_auc / total_valid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"heatmap_size: {args.heatmap_size}, heatmap_sigma: {args.heatmap_sigma}")
    print(f"centre gaussian sigma: {args.center_sigma} px")
    print()

    # data loaders — train for density estimation, val for evaluation
    train_loader = make_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=args.max_images,
        use_grouped_sampler=False,
        max_seq_len=args.max_seq_len,
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
        max_seq_len=args.max_seq_len,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    # build baseline heatmaps
    print("building baseline heatmaps...")
    uniform_log    = make_uniform_log_heatmap(args.heatmap_size, device)
    center_log     = make_center_gaussian_log_heatmap(args.heatmap_size, args.center_sigma, device)
    empirical_log  = estimate_empirical_density(
        train_loader, args.heatmap_size, args.heatmap_sigma, device, args.density_batches
    )
    print()

    # evaluate all baselines
    baselines = [
        ("uniform",           uniform_log),
        ("center_gaussian",   center_log),
        ("empirical_density", empirical_log),
    ]

    results: dict[str, dict] = {}
    for name, log_hm in baselines:
        print(f"evaluating: {name} ...", flush=True)
        m = evaluate_baseline(
            log_hm, val_loader, device,
            max_batches=args.eval_batches,
            heatmap_size=args.heatmap_size,
            heatmap_sigma=args.heatmap_sigma,
        )
        results[name] = m
        print(f"  steps={int(m['steps'])}  kl={m['kl']:.4f}  "
              f"ll={m['ll']:.4f}  nss={m['nss']:.4f}  auc={m['auc']:.4f}", flush=True)
        print()

    # summary table
    print("=" * 70)
    print(f"{'baseline':<22} {'KL':>8} {'LL':>8} {'NSS':>8} {'AUC':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<22} {m['kl']:>8.4f} {m['ll']:>8.4f} "
              f"{m['nss']:>8.4f} {m['auc']:>8.4f}")
    print("=" * 70)
    print()
    print("note: model val NSS from training logs (for reference):")
    print("  50k σ=2 (best epoch ~8): NSS ≈ 3.35")
    print("  200k σ=2 (epoch 5):      NSS ≈ 4.08  [still improving]")


if __name__ == "__main__":
    main()
