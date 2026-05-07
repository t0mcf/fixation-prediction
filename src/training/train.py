"""main training script for the scanpath model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel
from src.training.losses import kl_loss, nll_loss
from src.training.metrics import nss_score


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--parquet-path", default=default_parquet_path)
    parser.add_argument("--imagenet-root", default=default_imagenet_root)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # target heatmaps
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    # model
    parser.add_argument("--visual-encoder", default="dinov2_base")
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)

    # optimization
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # logging / validation / checkpointing
    parser.add_argument("--output-dir", default="runs/train")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)

    # wandb
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fixation-prediction")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-dir", default="wandb")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--image-log-every", type=int, default=500)

    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None

    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )

    return run


def normalize_for_image(x: torch.Tensor) -> torch.Tensor:
    """
    x: (H, W)
    returns: (H, W), normalized to [0, 1]
    """
    x = x.detach().float().cpu()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def log_wandb_heatmaps(
    run,
    log_pred: torch.Tensor,
    heatmap: torch.Tensor,
    step: int,
) -> None:
    if run is None:
        return

    import wandb

    pred = log_pred[0].exp()
    target = heatmap[0]

    pred_img = normalize_for_image(pred).numpy()
    target_img = normalize_for_image(target).numpy()

    run.log(
        {
            "sample/pred_heatmap": wandb.Image(pred_img, caption="predicted heatmap"),
            "sample/target_heatmap": wandb.Image(target_img, caption="target heatmap"),
        },
        step=step,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()

    total_kl = 0.0
    total_nll = 0.0
    total_nss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        image = batch["image"].to(device)
        prefix = batch["prefix"].to(device)
        prefix_len = batch["prefix_len"].to(device)
        heatmap = batch["heatmap"].to(device)
        target_xy = batch["target_xy"].to(device)

        log_pred = model(image, prefix, prefix_len)

        batch_size = image.shape[0]

        value_kl = kl_loss(log_pred, heatmap)
        value_nll = nll_loss(log_pred, target_xy)
        value_nss = nss_score(log_pred, target_xy)

        total_kl += value_kl.item() * batch_size
        total_nll += value_nll.item() * batch_size
        total_nss += value_nss.item() * batch_size
        total_samples += batch_size

    model.train()

    if total_samples == 0:
        return {
            "val_kl": float("nan"),
            "val_nll": float("nan"),
            "val_nss": float("nan"),
            "val_samples": 0,
        }

    return {
        "val_kl": total_kl / total_samples,
        "val_nll": total_nll / total_samples,
        "val_nss": total_nss / total_samples,
        "val_samples": total_samples,
    }


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "config.json", vars(args))

    wandb_run = init_wandb(args)

    train_loader = make_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=args.max_images,
        use_grouped_sampler=True,
        paths_per_image=16,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
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
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    model = ScanpathModel(
        visual_encoder=args.visual_encoder,
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

    log_path = output_dir / "log.csv"

    model.train()
    step = 0

    best_val_kl = float("inf")
    best_val_nss = float("-inf")

    print("starting training...", flush=True)

    try:
        while step < args.max_steps:
            for batch in train_loader:
                image = batch["image"].to(device)
                prefix = batch["prefix"].to(device)
                prefix_len = batch["prefix_len"].to(device)
                heatmap = batch["heatmap"].to(device)
                target_xy = batch["target_xy"].to(device)

                optimizer.zero_grad(set_to_none=True)

                log_pred = model(image, prefix, prefix_len)

                loss_kl = kl_loss(log_pred, heatmap)
                loss_nll = nll_loss(log_pred, target_xy)

                # optimize KL for first experiments, log NLL as diagnostic.
                loss = loss_kl

                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip,
                )

                optimizer.step()

                if step % args.log_every == 0:
                    train_row = {
                        "step": step,
                        "split": "train",
                        "kl": loss_kl.detach().item(),
                        "nll": loss_nll.detach().item(),
                        "nss": "",
                        "samples": image.shape[0],
                    }

                    append_csv(log_path, train_row)

                    lr = optimizer.param_groups[0]["lr"]

                    print(
                        f"step {step:06d} | "
                        f"train kl={train_row['kl']:.4f} | "
                        f"train nll={train_row['nll']:.4f} | "
                        f"lr={lr:.2e}",
                        flush=True,
                    )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/kl": train_row["kl"],
                                "train/nll": train_row["nll"],
                                "train/lr": lr,
                                "train/grad_norm": float(grad_norm),
                            },
                            step=step,
                        )

                if wandb_run is not None and step % args.image_log_every == 0:
                    log_wandb_heatmaps(
                        run=wandb_run,
                        log_pred=log_pred,
                        heatmap=heatmap,
                        step=step,
                    )

                if step > 0 and step % args.val_every == 0:
                    metrics = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        max_batches=args.val_batches,
                    )

                    val_row = {
                        "step": step,
                        "split": "val",
                        "kl": metrics["val_kl"],
                        "nll": metrics["val_nll"],
                        "nss": metrics["val_nss"],
                        "samples": metrics["val_samples"],
                    }

                    append_csv(log_path, val_row)

                    print(
                        f"step {step:06d} | "
                        f"val kl={val_row['kl']:.4f} | "
                        f"val nll={val_row['nll']:.4f} | "
                        f"val nss={val_row['nss']:.4f}",
                        flush=True,
                    )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/kl": val_row["kl"],
                                "val/nll": val_row["nll"],
                                "val/nss": val_row["nss"],
                                "val/samples": val_row["samples"],
                            },
                            step=step,
                        )

                    if metrics["val_kl"] < best_val_kl:
                        best_val_kl = metrics["val_kl"]
                        best_path = checkpoint_dir / "best_val_kl.pt"

                        save_checkpoint(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            args=args,
                        )

                        print(
                            f"saved new best val kl checkpoint: {best_path} "
                            f"(val_kl={best_val_kl:.4f})",
                            flush=True,
                        )

                        if wandb_run is not None:
                            wandb_run.log(
                                {"best/val_kl": best_val_kl},
                                step=step,
                            )

                    if metrics["val_nss"] > best_val_nss:
                        best_val_nss = metrics["val_nss"]
                        best_path = checkpoint_dir / "best_val_nss.pt"

                        save_checkpoint(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            args=args,
                        )

                        print(
                            f"saved new best val nss checkpoint: {best_path} "
                            f"(val_nss={best_val_nss:.4f})",
                            flush=True,
                        )

                        if wandb_run is not None:
                            wandb_run.log(
                                {"best/val_nss": best_val_nss},
                                step=step,
                            )

                if step > 0 and step % args.save_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        args=args,
                    )
                    print(f"saved checkpoint: {ckpt_path}", flush=True)

                step += 1
                if step >= args.max_steps:
                    break

        final_path = checkpoint_dir / "final.pt"
        save_checkpoint(
            path=final_path,
            model=model,
            optimizer=optimizer,
            step=step,
            args=args,
        )

        print(f"saved final checkpoint: {final_path}", flush=True)
        print(f"best val kl: {best_val_kl:.4f}", flush=True)
        print(f"best val nss: {best_val_nss:.4f}", flush=True)
        print("training finished", flush=True)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/step": step,
                    "final/best_val_kl": best_val_kl,
                    "final/best_val_nss": best_val_nss,
                },
                step=step,
            )

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()