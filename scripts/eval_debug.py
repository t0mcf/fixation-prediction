"""small debug evaluation script for the scanpath model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel


def kl_loss(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    log_pred: (B, H, W)
    target: (B, H, W)
    """
    B = log_pred.shape[0]

    log_pred_flat = log_pred.view(B, -1)
    target_flat = target.view(B, -1)
    log_target_flat = torch.log(target_flat.clamp_min(1e-8))

    return F.kl_div(
        log_pred_flat,
        log_target_flat,
        reduction="batchmean",
        log_target=True,
    )


def nll_loss(log_pred: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    log_pred: (B, H, W)
    target_xy: (B, 2), normalized coordinates in [-1, 1]
    """
    B, H, W = log_pred.shape
    device = log_pred.device

    x = target_xy[:, 0]
    y = target_xy[:, 1]

    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    log_p = log_pred[torch.arange(B, device=device), row, col]

    return -log_p.mean()


def nss_score(log_pred: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    log_pred: (B, H, W), log-probabilities
    target_xy: (B, 2), normalized coordinates in [-1, 1]

    returns:
        mean NSS over the batch
    """
    pred = log_pred.exp()
    B, H, W = pred.shape
    device = pred.device

    x = target_xy[:, 0]
    y = target_xy[:, 1]

    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    pred_flat = pred.view(B, -1)
    mean = pred_flat.mean(dim=1).view(B, 1, 1)
    std = pred_flat.std(dim=1).view(B, 1, 1).clamp_min(1e-8)

    pred_norm = (pred - mean) / std
    values = pred_norm[torch.arange(B, device=device), row, col]

    return values.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet-path", default="data/debug_scanpaths.parquet")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)

    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model = ScanpathModel(
        visual_encoder="dinov2_base",
        pretrained_encoder=not ckpt_args.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=ckpt_args.get("d_model", 256),
        n_heads=ckpt_args.get("n_heads", 8),
        scanpath_layers=ckpt_args.get("scanpath_layers", 2),
        max_prefix_len=15,
        heatmap_size=ckpt_args.get("heatmap_size", args.heatmap_size),
        dropout=ckpt_args.get("dropout", 0.1),
        decoder_hidden_channels=ckpt_args.get("decoder_hidden_channels", 256),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        use_grouped_sampler=False,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        parquet_path=args.parquet_path,
    )

    total_kl = 0.0
    total_nll = 0.0
    total_nss = 0.0
    total_samples = 0

    print("starting debug evaluation...", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            image = batch["image"].to(device)
            prefix = batch["prefix"].to(device)
            prefix_len = batch["prefix_len"].to(device)
            heatmap = batch["heatmap"].to(device)
            target_xy = batch["target_xy"].to(device)

            log_pred = model(image, prefix, prefix_len)

            batch_size = image.shape[0]

            loss_kl = kl_loss(log_pred, heatmap)
            loss_nll = nll_loss(log_pred, target_xy)
            score_nss = nss_score(log_pred, target_xy)

            total_kl += loss_kl.item() * batch_size
            total_nll += loss_nll.item() * batch_size
            total_nss += score_nss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 5 == 0:
                print(
                    f"batch {batch_idx:04d} | "
                    f"kl={loss_kl.item():.4f} | "
                    f"nll={loss_nll.item():.4f} | "
                    f"nss={score_nss.item():.4f}",
                    flush=True,
                )

    mean_kl = total_kl / total_samples
    mean_nll = total_nll / total_samples
    mean_nss = total_nss / total_samples

    print("\nsummary", flush=True)
    print(f"samples: {total_samples}", flush=True)
    print(f"kl:      {mean_kl:.4f}", flush=True)
    print(f"nll:     {mean_nll:.4f}", flush=True)
    print(f"nss:     {mean_nss:.4f}", flush=True)


if __name__ == "__main__":
    main()