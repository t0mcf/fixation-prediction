"""small debug training run for the scanpath model."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--parquet-path", default="data/debug_scanpaths.parquet")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)

    parser.add_argument(
        "--no-pretrained-encoder",
        action="store_true",
        help="use randomly initialized visual encoder weights for quick shape/debug tests",
    )

    parser.add_argument("--output-dir", default="runs/debug")
    parser.add_argument("--save-checkpoint", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        use_grouped_sampler=False,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        parquet_path=args.parquet_path,
    )

    model = ScanpathModel(
        visual_encoder="dinov2_base",
        pretrained_encoder=not args.no_pretrained_encoder,
        img_size=224,
        d_model=args.d_model,
        n_heads=args.n_heads,
        scanpath_layers=args.scanpath_layers,
        max_prefix_len=15,
        heatmap_size=args.heatmap_size,
        dropout=args.dropout,
        decoder_hidden_channels=args.decoder_hidden_channels,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()

    step = 0
    print("starting debug training...", flush=True)

    while step < args.max_steps:
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

            # for the first experiments we optimize KL to stay comparable
            # to the previous prototype, while also monitoring point NLL.
            loss = loss_kl

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
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
            if step >= args.max_steps:
                break

    print("debug training finished", flush=True)

    if args.save_checkpoint:
        ckpt_path = output_dir / "checkpoint_debug.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"saved checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()