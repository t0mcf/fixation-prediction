"""small debug training run for the scanpath model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW


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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    loader = make_dataloader(
        split="train",
        batch_size=2,
        num_workers=0,
        max_images=5,
        use_grouped_sampler=False,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )

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

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    model.train()

    max_steps = 50
    step = 0

    print("starting debug training...", flush=True)

    while step < max_steps:
        for batch in loader:
            image = batch["image"].to(device)
            prefix = batch["prefix"].to(device)
            prefix_len = batch["prefix_len"].to(device)
            heatmap = batch["heatmap"].to(device)
            target_xy = batch["target_xy"].to(device)

            optimizer.zero_grad(set_to_none=True)

            log_pred = model(image, prefix, prefix_len)

            loss_kl = kl_loss(log_pred, heatmap)
            loss_nll = nll_loss(log_pred, target_xy)

            loss = loss_kl

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            optimizer.step()

            if step % 5 == 0:
                probs_sum = log_pred.exp().sum(dim=(-1, -2)).detach().cpu()

                print(
                    f"step {step:04d} | "
                    f"kl={loss_kl.detach().item():.4f} | "
                    f"nll={loss_nll.detach().item():.4f} | "
                    f"prob_sum={probs_sum.mean().item():.4f}",
                    flush=True,
                )

            step += 1
            if step >= max_steps:
                break

    print("debug training finished", flush=True)


if __name__ == "__main__":
    main()