"""main training script for the scanpath model (predict-all formulation)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel
from src.training.losses import (
    make_gaussian_heatmaps_batch,
    multi_step_kl_loss,
    multi_step_ll_score,
    kl_loss,
    ll_score,
)
from src.training.metrics import nss_score, auc_score


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
    parser.add_argument("--paths-per-image", type=int, default=16)
    parser.add_argument(
        "--ignore-prefix",
        action="store_true",
        help="ignore scanpath history (pure visual saliency ablation)",
    )

    # target heatmaps
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=8.0)

    # model
    parser.add_argument("--visual-encoder", default="dinov2_base")
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=16,
                        help="maximum total scanpath length (full sequence including first fixation)")

    # optimization
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.1,
        help="lr multiplier for the pretrained encoder relative to --lr (only active if encoder is unfrozen)",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default="cosine",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="explicit warmup step count; if 0, --warmup-fraction is used instead",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.05,
        help="fraction of total steps used for lr warmup when --warmup-steps is 0",
    )
    parser.add_argument("--amp", action="store_true", help="use automatic mixed precision")

    # logging / validation / checkpointing
    parser.add_argument("--output-dir", default="runs/train")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=100,
                        help="validation batches per epoch; -1 = full val set")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")

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
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    scaler: torch.amp.GradScaler | None = None,
    best_val_kl: float | None = None,
    best_val_nss: float | None = None,
) -> None:
    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    if best_val_kl is not None:
        ckpt["best_val_kl"] = best_val_kl
    if best_val_nss is not None:
        ckpt["best_val_nss"] = best_val_nss
    torch.save(ckpt, path)


def init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    import wandb
    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )


def log_wandb_heatmaps(run, log_preds, target_heatmaps, step):
    """Log the first sample's first prediction step for visual inspection."""
    import wandb
    pred = log_preds[0, 0].exp().detach().float().cpu().numpy()
    target = target_heatmaps[0, 0].detach().float().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    import numpy as np
    overlay = np.stack([pred, target, np.zeros_like(pred)], axis=-1)
    run.log({
        "sample/pred": wandb.Image(pred, caption="pred"),
        "sample/target": wandb.Image(target, caption="target"),
        "sample/overlay": wandb.Image(overlay, caption="red=pred, green=target"),
    }, step=step)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    use_amp: bool,
    ignore_prefix: bool,
    heatmap_size: int,
    heatmap_sigma: float,
) -> dict[str, float]:
    model.eval()
    device_type = device.split(":")[0]

    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        image = batch["image"].to(device)
        fixations = batch["fixations"].to(device)
        fixations_len = batch["fixations_len"].to(device)

        B, max_seq_len, _ = fixations.shape
        T = max_seq_len - 1

        target_fixations = fixations[:, 1:, :]  # (B, T, 2)
        target_heatmaps = make_gaussian_heatmaps_batch(
            target_fixations, heatmap_size, heatmap_sigma
        ).float()

        with torch.autocast(device_type=device_type, enabled=use_amp):
            log_preds = model(image, fixations, fixations_len, ignore_prefix)

        log_preds = log_preds.float()

        # valid_mask[b, t] = True when step t has a real prediction target
        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < (fixations_len - 1).unsqueeze(1)
        )  # (B, T)

        # flatten valid (sample, step) pairs for per-step metrics
        valid_flat = valid_mask.view(-1)
        if not valid_flat.any():
            continue

        lp = log_preds.view(B * T, *log_preds.shape[2:])[valid_flat]       # (V, H, W)
        th = target_heatmaps.view(B * T, *target_heatmaps.shape[2:])[valid_flat]  # (V, H, W)
        txy = target_fixations.reshape(B * T, 2)[valid_flat]                # (V, 2)
        n_valid = valid_flat.sum().item()

        total_kl  += kl_loss(lp, th).item() * n_valid
        total_ll  += ll_score(lp, txy).item() * n_valid
        total_nss += nss_score(lp, txy).item() * n_valid
        total_auc += auc_score(lp, txy).item() * n_valid
        total_valid += n_valid

    model.train()

    if total_valid == 0:
        return {"val_kl": float("nan"), "val_ll": float("nan"),
                "val_nss": float("nan"), "val_auc": float("nan"), "val_steps": 0}

    return {
        "val_kl":   total_kl   / total_valid,
        "val_ll":   total_ll   / total_valid,
        "val_nss":  total_nss  / total_valid,
        "val_auc":  total_auc  / total_valid,
        "val_steps": total_valid,
    }


def run_validation(
    model, val_loader, optimizer, scheduler, scaler,
    device, step, epoch, epoch_float, use_amp, args,
    log_path, checkpoint_dir, wandb_run, best_val_kl, best_val_nss,
) -> tuple[float, float]:
    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.val_batches,
        use_amp=use_amp,
        ignore_prefix=args.ignore_prefix,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
    )

    val_row = {
        "step": step, "epoch": epoch_float, "split": "val",
        "kl": metrics["val_kl"], "ll": metrics["val_ll"],
        "nss": metrics["val_nss"], "auc": metrics["val_auc"],
        "steps": metrics["val_steps"],
    }
    append_csv(log_path, val_row)

    print(
        f"epoch {epoch + 1:03d} | step {step:06d} | "
        f"val kl={val_row['kl']:.4f} | val ll={val_row['ll']:.4f} | "
        f"val nss={val_row['nss']:.4f} | val auc={val_row['auc']:.4f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.log(
            {"val/kl": val_row["kl"], "val/ll": val_row["ll"],
             "val/nss": val_row["nss"], "val/auc": val_row["auc"]},
            step=step,
        )

    if metrics["val_kl"] < best_val_kl:
        best_val_kl = metrics["val_kl"]
        best_path = checkpoint_dir / "best_val_kl.pt"
        save_checkpoint(best_path, model, optimizer, scheduler, step, epoch, args,
                        scaler, best_val_kl, best_val_nss)
        print(f"saved new best val kl checkpoint (val_kl={best_val_kl:.4f})", flush=True)

    if metrics["val_nss"] > best_val_nss:
        best_val_nss = metrics["val_nss"]
        best_path = checkpoint_dir / "best_val_nss.pt"
        save_checkpoint(best_path, model, optimizer, scheduler, step, epoch, args,
                        scaler, best_val_kl, best_val_nss)
        print(f"saved new best val nss checkpoint (val_nss={best_val_nss:.4f})", flush=True)

    return best_val_kl, best_val_nss


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device.split(":")[0]

    use_amp = args.amp and device == "cuda"
    if args.amp and not use_amp:
        print("warning: --amp requested but device is not cuda, disabling", flush=True)

    print(f"device: {device}", flush=True)
    print(f"amp: {use_amp}", flush=True)
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
        paths_per_image=args.paths_per_image,
        max_seq_len=args.max_seq_len,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=None,
        use_grouped_sampler=False,
        max_seq_len=args.max_seq_len,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch

    print(f"train batches per epoch: {steps_per_epoch}", flush=True)
    print(f"planned epochs: {args.num_epochs}", flush=True)
    print(f"planned steps: {total_steps}", flush=True)

    model = ScanpathModel(
        visual_encoder=args.visual_encoder,
        pretrained_encoder=not args.no_pretrained_encoder,
        img_size=224,
        d_model=args.d_model,
        n_heads=args.n_heads,
        scanpath_layers=args.scanpath_layers,
        fusion_layers=args.fusion_layers,
        max_seq_len=args.max_seq_len,
        heatmap_size=args.heatmap_size,
        dropout=args.dropout,
        decoder_hidden_channels=args.decoder_hidden_channels,
    ).to(device)

    # the visual encoder is frozen, so only non-encoder params are trainable.
    # encoder_lr_scale is reserved for future fine-tuning experiments.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = (
        args.warmup_steps if args.warmup_steps > 0
        else int(total_steps * args.warmup_fraction)
    )

    scheduler = make_scheduler(optimizer, warmup_steps, total_steps, args.lr_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_path = output_dir / "log.csv"

    start_epoch = 0
    step = 0
    best_val_kl = float("inf")
    best_val_nss = float("-inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        best_val_kl = ckpt.get("best_val_kl", float("inf"))
        best_val_nss = ckpt.get("best_val_nss", float("-inf"))
        print(f"resumed from {args.resume} | start_epoch={start_epoch} | step={step}", flush=True)

    model.train()
    print("starting training...", flush=True)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            model.train()

            epoch_loss_kl = 0.0
            epoch_loss_ll = 0.0
            epoch_valid_steps = 0

            print(f"starting epoch {epoch + 1}/{args.num_epochs}", flush=True)

            for batch_idx, batch in enumerate(train_loader):
                image = batch["image"].to(device)
                fixations = batch["fixations"].to(device)       # (B, max_seq_len, 2)
                fixations_len = batch["fixations_len"].to(device)  # (B,)

                # targets: fixation[t+1] is the prediction target for step t
                target_fixations = fixations[:, 1:, :]  # (B, T, 2)
                target_heatmaps = make_gaussian_heatmaps_batch(
                    target_fixations, args.heatmap_size, args.heatmap_sigma
                ).float()

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device_type, enabled=use_amp):
                    log_preds = model(image, fixations, fixations_len, args.ignore_prefix)

                log_preds = log_preds.float()

                loss_kl = multi_step_kl_loss(log_preds, target_heatmaps, fixations_len)
                loss_ll = multi_step_ll_score(log_preds, target_fixations, fixations_len)

                if not torch.isfinite(loss_kl):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_kl.item()}")

                scaler.scale(loss_kl).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.grad_clip, error_if_nonfinite=True,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1

                B = image.shape[0]
                T = fixations.shape[1] - 1
                valid_steps_in_batch = (
                    torch.arange(T, device=device).unsqueeze(0)
                    < (fixations_len - 1).unsqueeze(1)
                ).sum().item()

                epoch_loss_kl += loss_kl.detach().item() * valid_steps_in_batch
                epoch_loss_ll += loss_ll.detach().item() * valid_steps_in_batch
                epoch_valid_steps += valid_steps_in_batch

                epoch_float = epoch + batch_idx / max(steps_per_epoch, 1)

                if step % args.log_every == 0:
                    train_row = {
                        "step": step, "epoch": epoch_float, "split": "train",
                        "kl": loss_kl.detach().item(),
                        "ll": loss_ll.detach().item(),
                        "nss": "", "auc": "",
                        "valid_steps": valid_steps_in_batch,
                    }
                    append_csv(log_path, train_row)

                    lr = scheduler.get_last_lr()[0]

                    print(
                        f"epoch {epoch + 1:03d}/{args.num_epochs:03d} | "
                        f"step {step:06d} | "
                        f"train kl={train_row['kl']:.4f} | "
                        f"train ll={train_row['ll']:.4f} | "
                        f"lr={lr:.2e}",
                        flush=True,
                    )

                    if wandb_run is not None:
                        module_grad_norms = {}
                        for name, module in [
                            ("scanpath_encoder", model.scanpath_encoder),
                            ("fusion", model.fusion),
                            ("decoder", model.decoder),
                            ("image_adapter", model.image_adapter),
                        ]:
                            total_norm = sum(
                                p.grad.detach().norm().item() ** 2
                                for p in module.parameters() if p.grad is not None
                            ) ** 0.5
                            module_grad_norms[f"grad_norm/{name}"] = total_norm

                        wandb_run.log(
                            {
                                "train/kl": train_row["kl"],
                                "train/ll": train_row["ll"],
                                "train/lr": lr,
                                "train/grad_norm": float(grad_norm),
                                **module_grad_norms,
                            },
                            step=step,
                        )


                if wandb_run is not None and step % args.image_log_every == 0:
                    log_wandb_heatmaps(wandb_run, log_preds, target_heatmaps, step)

            mean_epoch_kl = epoch_loss_kl / max(epoch_valid_steps, 1)
            mean_epoch_ll = epoch_loss_ll / max(epoch_valid_steps, 1)

            print(
                f"finished epoch {epoch + 1}/{args.num_epochs} | "
                f"mean train kl={mean_epoch_kl:.4f} | mean train ll={mean_epoch_ll:.4f}",
                flush=True,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {"train/epoch_kl": mean_epoch_kl, "train/epoch_ll": mean_epoch_ll},
                    step=step,
                )

            best_val_kl, best_val_nss = run_validation(
                model=model, val_loader=val_loader, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, device=device,
                step=step, epoch=epoch, epoch_float=float(epoch + 1),
                use_amp=use_amp, args=args, log_path=log_path,
                checkpoint_dir=checkpoint_dir, wandb_run=wandb_run,
                best_val_kl=best_val_kl, best_val_nss=best_val_nss,
            )

            latest_path = checkpoint_dir / "latest.pt"
            save_checkpoint(latest_path, model, optimizer, scheduler,
                            step, epoch, args, scaler, best_val_kl, best_val_nss)
            print(f"saved latest checkpoint: {latest_path}", flush=True)

        final_path = checkpoint_dir / "final.pt"
        final_epoch = epoch if "epoch" in locals() else start_epoch - 1
        save_checkpoint(final_path, model, optimizer, scheduler,
                        step, final_epoch, args, scaler, best_val_kl, best_val_nss)
        print(f"saved final checkpoint: {final_path}", flush=True)
        print(f"total steps: {step}", flush=True)
        print(f"best val kl: {best_val_kl:.4f}", flush=True)
        print(f"best val nss: {best_val_nss:.4f}", flush=True)
        print("training finished", flush=True)

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
