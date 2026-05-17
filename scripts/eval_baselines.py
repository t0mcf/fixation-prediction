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
from src.training.losses import make_gaussian_heatmaps_batch, kl_loss, ll_score
from src.training.metrics import nss_score, auc_score


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def make_uniform_log_pred(batch_size: int, heatmap_size: int, device: str) -> torch.Tensor:
    H = W = heatmap_size
    logits = torch.zeros(batch_size, H, W, device=device)
    return F.log_softmax(logits.view(batch_size, -1), dim=-1).view(batch_size, H, W)


def make_center_gaussian_log_pred(
    batch_size: int,
    heatmap_size: int,
    sigma: float,
    device: str,
) -> torch.Tensor:
    H = W = heatmap_size
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0
    logits = -((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma ** 2)
    logits = logits.unsqueeze(0).expand(batch_size, H, W)
    return F.log_softmax(logits.view(batch_size, -1), dim=-1).view(batch_size, H, W)


def build_empirical_density(
    loader: torch.utils.data.DataLoader,
    heatmap_size: int,
    device: str,
    max_batches: int | None = None,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Build empirical fixation density from all target fixations in training data."""
    counts = torch.full(
        (heatmap_size, heatmap_size),
        fill_value=smoothing,
        dtype=torch.float32,
        device=device,
    )

    H = W = heatmap_size

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        fixations = batch["fixations"].to(device)      # (B, max_seq_len, 2)
        fixations_len = batch["fixations_len"].to(device)  # (B,)
        B, max_seq_len, _ = fixations.shape
        T = max_seq_len - 1

        # accumulate all valid target fixations (fixations[1..N-1] for each sample)
        for b in range(B):
            n_valid = fixations_len[b].item() - 1  # number of valid prediction steps
            if n_valid <= 0:
                continue
            targets = fixations[b, 1:1 + int(n_valid), :]  # (n_valid, 2)
            x = targets[:, 0]
            y = targets[:, 1]
            col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
            row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)
            for r, c in zip(row, col):
                counts[r, c] += 1.0

    probs = counts / counts.sum()
    return torch.log(probs.clamp_min(1e-8))


def evaluate_baseline(
    loader: torch.utils.data.DataLoader,
    name: str,
    make_log_pred,
    device: str,
    max_batches: int,
    heatmap_size: int,
    heatmap_sigma: float,
) -> dict[str, float]:
    """Evaluate a baseline over all valid (sample, step) pairs."""
    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            fixations = batch["fixations"].to(device)       # (B, max_seq_len, 2)
            fixations_len = batch["fixations_len"].to(device)  # (B,)
            B, max_seq_len, _ = fixations.shape
            T = max_seq_len - 1

            target_fixations = fixations[:, 1:, :]  # (B, T, 2)
            target_heatmaps = make_gaussian_heatmaps_batch(
                target_fixations, heatmap_size, heatmap_sigma
            )  # (B, T, H, W)

            valid_mask = (
                torch.arange(T, device=device).unsqueeze(0)
                < (fixations_len - 1).unsqueeze(1)
            )  # (B, T)

            valid_flat = valid_mask.view(-1)
            if not valid_flat.any():
                continue

            # baseline prediction is the same for every step — shape (B, H, W)
            log_pred = make_log_pred(B)  # (B, H, W)
            # expand to (B, T, H, W) then flatten valid pairs
            log_pred_expanded = log_pred.unsqueeze(1).expand(B, T, -1, -1)

            lp = log_pred_expanded.reshape(B * T, *log_pred.shape[1:])[valid_flat]
            th = target_heatmaps.view(B * T, *target_heatmaps.shape[2:])[valid_flat]
            txy = target_fixations.reshape(B * T, 2)[valid_flat]
            n_valid = valid_flat.sum().item()

            total_kl  += kl_loss(lp, th).item() * n_valid
            total_ll  += ll_score(lp, txy).item() * n_valid
            total_nss += nss_score(lp, txy).item() * n_valid
            total_auc += auc_score(lp, txy).item() * n_valid
            total_valid += n_valid

    if total_valid == 0:
        return {"name": name, "steps": 0,
                "kl": float("nan"), "ll": float("nan"),
                "nss": float("nan"), "auc": float("nan")}

    return {
        "name": name,
        "steps": total_valid,
        "kl":  total_kl  / total_valid,
        "ll":  total_ll  / total_valid,
        "nss": total_nss / total_valid,
        "auc": total_auc / total_valid,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--parquet-path", default=default_parquet_path)
    parser.add_argument("--imagenet-root", default=default_imagenet_root)
    parser.add_argument("--max-images", type=int, default=None,
                        help="limit val set by image count; None = full val set")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=16)

    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=8.0)
    parser.add_argument("--center-sigma", type=float, default=12.0)
    parser.add_argument("--density-smoothing", type=float, default=1.0)
    parser.add_argument("--density-batches", type=int, default=None,
                        help="number of train batches for building empirical density; None = all")
    parser.add_argument("--eval-batches", type=int, default=200)

    return parser.parse_args()


def print_metrics(metrics: dict[str, float]) -> None:
    print(
        f"{metrics['name']:>20} | "
        f"steps={int(metrics['steps']):6d} | "
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

    print("building empirical train density...", flush=True)
    train_log_density = build_empirical_density(
        loader=train_loader,
        heatmap_size=args.heatmap_size,
        device=device,
        max_batches=args.density_batches,
        smoothing=args.density_smoothing,
    )

    print("evaluating baselines on val set...", flush=True)
    H = W = args.heatmap_size

    baselines = [
        evaluate_baseline(
            loader=val_loader,
            name="uniform",
            make_log_pred=lambda B: make_uniform_log_pred(B, args.heatmap_size, device),
            device=device,
            max_batches=args.eval_batches,
            heatmap_size=args.heatmap_size,
            heatmap_sigma=args.heatmap_sigma,
        ),
        evaluate_baseline(
            loader=val_loader,
            name="center_gaussian",
            make_log_pred=lambda B: make_center_gaussian_log_pred(
                B, args.heatmap_size, args.center_sigma, device
            ),
            device=device,
            max_batches=args.eval_batches,
            heatmap_size=args.heatmap_size,
            heatmap_sigma=args.heatmap_sigma,
        ),
        evaluate_baseline(
            loader=val_loader,
            name="empirical_density",
            make_log_pred=lambda B: train_log_density.unsqueeze(0).expand(B, H, W),
            device=device,
            max_batches=args.eval_batches,
            heatmap_size=args.heatmap_size,
            heatmap_sigma=args.heatmap_sigma,
        ),
    ]

    print("\nsummary", flush=True)
    for metrics in baselines:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
