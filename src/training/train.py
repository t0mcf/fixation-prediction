"""main training script for the scanpath model."""

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
from src.training.losses import kl_loss, ll_score
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
    help="ignore scanpath history by zeroing prefix and setting prefix_len to 0",
    )

    # target heatmaps
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    # model
    parser.add_argument("--visual-encoder", default="dinov2_base")
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)
    parser.add_argument("--max-prefix-len", type=int, default=15)

    # optimization
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.1,
        help="lr multiplier for the pretrained encoder relative to --lr",
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
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--resume", default=None, help="path to an epoch checkpoint to resume from")

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


def log_wandb_heatmaps(run, log_pred, heatmap, step):
    import wandb
    import numpy as np

    pred = log_pred[0].exp().detach().float().cpu().numpy()
    target = heatmap[0].detach().float().cpu().numpy()

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

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
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # +1 so the very first step gets lr/warmup_steps instead of 0
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    use_amp: bool = False,
    ignore_prefix: bool = False,
) -> dict[str, float]:
    model.eval()

    device_type = device.split(":")[0] #in case it is e.g. cuda:0 
    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        image = batch["image"].to(device)
        prefix = batch["prefix"].to(device)
        prefix_len = batch["prefix_len"].to(device)
        
        heatmap = batch["heatmap"].to(device)
        target_xy = batch["target_xy"].to(device)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            log_pred = model(image, prefix, prefix_len, ignore_prefix)
            batch_size = image.shape[0]

        value_kl = kl_loss(log_pred, heatmap)
        value_ll = ll_score(log_pred, target_xy)
        value_nss = nss_score(log_pred, target_xy)
        value_auc = auc_score(log_pred, target_xy)

        total_kl += value_kl.item() * batch_size # weight with batch_size since last batch may be smaller
        total_ll += value_ll.item() * batch_size
        total_nss += value_nss.item() * batch_size
        total_auc += value_auc.item() * batch_size
        total_samples += batch_size

    model.train()

    if total_samples == 0:
        return {
            "val_kl": float("nan"),
            "val_ll": float("nan"),
            "val_nss": float("nan"),
            "val_auc": float("nan"),
            "val_samples": 0,
        }

    return {
        "val_kl": total_kl / total_samples,
        "val_ll": total_ll / total_samples,
        "val_nss": total_nss / total_samples,
        "val_auc": total_auc / total_samples,
        "val_samples": total_samples,
    }


def run_validation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: str,
    step: int,
    epoch: int,
    epoch_float: float,
    use_amp: bool,
    args: argparse.Namespace,
    log_path: Path,
    checkpoint_dir: Path,
    wandb_run,
    best_val_kl: float,
    best_val_nss: float,
) -> tuple[float, float]:
    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.val_batches,
        use_amp=use_amp,
        ignore_prefix=args.ignore_prefix,
    )

    val_row = {
        "step": step,
        "epoch": epoch_float,
        "split": "val",
        "kl": metrics["val_kl"],
        "ll": metrics["val_ll"],
        "nss": metrics["val_nss"],
        "auc": metrics["val_auc"],
        "samples": metrics["val_samples"],
    }

    append_csv(log_path, val_row)

    print(
        f"epoch {epoch + 1:03d}/{args.num_epochs:03d} | "
        f"step {step:06d} | "
        f"val kl={val_row['kl']:.4f} | "
        f"val ll={val_row['ll']:.4f} | "
        f"val nss={val_row['nss']:.4f}",
        f"val auc={val_row['auc']:.4f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "val/kl": val_row["kl"],
                "val/ll": val_row["ll"],
                "val/nss": val_row["nss"],
                "val/auc": val_row["auc"],
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
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            args=args,
            scaler=scaler,
            best_val_kl=best_val_kl,
            best_val_nss=best_val_nss,
        )

        print(
            f"saved new best val kl checkpoint: {best_path} "
            f"(val_kl={best_val_kl:.4f})",
            flush=True,
        )

    if metrics["val_nss"] > best_val_nss:
        best_val_nss = metrics["val_nss"]
        best_path = checkpoint_dir / "best_val_nss.pt"

        save_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            args=args,
            scaler=scaler,
            best_val_kl=best_val_kl,
            best_val_nss=best_val_nss,
        )

        print(
            f"saved new best val nss checkpoint: {best_path} "
            f"(val_nss={best_val_nss:.4f})",
            flush=True,
        )

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
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
        max_prefix_len=args.max_prefix_len,
    )

    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_images=None, #so val is independent of the number of images 
        use_grouped_sampler=False,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_size=args.heatmap_size,
        parquet_path=args.parquet_path,
        imagenet_root=args.imagenet_root,
        max_prefix_len=args.max_prefix_len,
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
        max_prefix_len=args.max_prefix_len,
        heatmap_size=args.heatmap_size,
        dropout=args.dropout,
        decoder_hidden_channels=args.decoder_hidden_channels,
    ).to(device)

    # lower lr for the pretrained encoder to avoid destroying pretrained features;
    # adjust model.image_encoder if the attribute name differs
    encoder_params = list(model.image_encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

    trainable_encoder = [p for p in encoder_params if p.requires_grad]
    trainable_other = [p for p in other_params if p.requires_grad]
    trainable_params = trainable_encoder + trainable_other

    optimizer = AdamW(
        [
            {"params": trainable_encoder, "lr": args.lr * args.encoder_lr_scale},
            {"params": trainable_other, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else int(total_steps * args.warmup_fraction)
    )

    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

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
        print(
            f"resumed from {args.resume} | "
            f"start_epoch={start_epoch} | step={step}",
            flush=True,
        )

    model.train()

    print("starting training...", flush=True)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            train_loader.sampler.set_epoch(epoch) # for shuffling with ImageGroupedSampler
            model.train()

            epoch_loss_kl = 0.0
            epoch_loss_ll = 0.0
            epoch_samples = 0

            print(f"starting epoch {epoch + 1}/{args.num_epochs}", flush=True)

            for batch_idx, batch in enumerate(train_loader):
                image = batch["image"].to(device)
                prefix = batch["prefix"].to(device)
                prefix_len = batch["prefix_len"].to(device)
                    
                heatmap = batch["heatmap"].to(device)
                target_xy = batch["target_xy"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device_type, enabled=use_amp):
                    log_pred = model(image, prefix, prefix_len, args.ignore_prefix)


                log_pred = log_pred.float()
                heatmap = heatmap.float()
                
                loss_kl = kl_loss(log_pred, heatmap)
                loss_ll = ll_score(log_pred, target_xy)
                loss = loss_kl #optimizing KL for now, ll just being logged
                
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

                scaler.scale(loss).backward()
                # unscale before clipping so we clip the actual gradients, not the scaled ones
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=args.grad_clip,
                    error_if_nonfinite=True,
                )
                
                if step % args.log_every == 0 and wandb_run is not None:
                    module_grad_norms = {}
                    for name, module in [
                        ("scanpath_encoder", model.scanpath_encoder),
                        ("fusion", model.fusion),
                        ("decoder", model.decoder),
                        ("image_adapter", model.image_adapter),
                    ]:
                        total_norm = 0.0
                        for p in module.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.detach().norm().item() ** 2
                        module_grad_norms[f"grad_norm/{name}"] = total_norm ** 0.5

                    wandb_run.log(module_grad_norms, step=step)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                step += 1

                batch_size = image.shape[0]
                epoch_float = epoch + batch_idx / max(steps_per_epoch, 1)

                epoch_loss_kl += loss_kl.detach().item() * batch_size
                epoch_loss_ll += loss_ll.detach().item() * batch_size
                epoch_samples += batch_size
                
                #wandb logging
                if step % args.log_every == 0:
                    train_row = {
                        "step": step,
                        "epoch": epoch_float,
                        "split": "train",
                        "kl": loss_kl.detach().item(),
                        "ll": loss_ll.detach().item(),
                        "nss": "",
                        "auc": "",
                        "samples": batch_size,
                    }

                    append_csv(log_path, train_row)

                    # index 1 = other_params group (the non-encoder lr)
                    lr = scheduler.get_last_lr()[1]
                    encoder_lr = scheduler.get_last_lr()[0]

                    module_grad_norms = {}
                    for name, module in [
                        ("scanpath_encoder", model.scanpath_encoder),
                        ("fusion", model.fusion),
                        ("decoder", model.decoder),
                        ("image_adapter", model.image_adapter),
                    ]:
                        total_norm = 0.0
                        for p in module.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.detach().norm().item() ** 2
                        module_grad_norms[f"grad_norm/{name}"] = total_norm ** 0.5

                    print(
                        f"epoch {epoch + 1:03d}/{args.num_epochs:03d} | "
                        f"step {step:06d} | "
                        f"train kl={train_row['kl']:.4f} | "
                        f"train ll={train_row['ll']:.4f} | "
                        f"lr={lr:.2e} | encoder_lr={encoder_lr:.2e}",
                        flush=True,
                    )

                    if wandb_run is not None:
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
                        
                #wandb heatmap logging
                if wandb_run is not None and step % args.image_log_every == 0:
                    log_wandb_heatmaps(
                        run=wandb_run,
                        log_pred=log_pred,
                        heatmap=heatmap,
                        step=step,
                    )

            #epoch end 
            mean_epoch_kl = epoch_loss_kl / max(epoch_samples, 1)
            mean_epoch_ll = epoch_loss_ll / max(epoch_samples, 1)

            print(
                f"finished epoch {epoch + 1}/{args.num_epochs} | "
                f"mean train kl={mean_epoch_kl:.4f} | "
                f"mean train ll={mean_epoch_ll:.4f}",
                flush=True,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch_kl": mean_epoch_kl,
                        "train/epoch_ll": mean_epoch_ll,
                    },
                    step=step,
                )

            # end-of-epoch validation
            best_val_kl, best_val_nss = run_validation(
                model=model,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                step=step,
                epoch=epoch,
                epoch_float=float(epoch + 1),
                use_amp=use_amp,
                args=args,
                log_path=log_path,
                checkpoint_dir=checkpoint_dir,
                wandb_run=wandb_run,
                best_val_kl=best_val_kl,
                best_val_nss=best_val_nss,
            )

            # epoch checkpoint used as resume boundary
            epoch_ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                path=epoch_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
                args=args,
                scaler=scaler,
                best_val_kl=best_val_kl,
                best_val_nss=best_val_nss,
            )
            print(f"saved epoch checkpoint: {epoch_ckpt_path}", flush=True)

        final_path = checkpoint_dir / "final.pt"
        final_epoch = epoch if "epoch" in locals() else start_epoch - 1
        save_checkpoint(
            path=final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=final_epoch,
            args=args,
            scaler=scaler,
            best_val_kl=best_val_kl,
            best_val_nss=best_val_nss,
        )

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