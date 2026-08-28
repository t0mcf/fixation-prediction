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
import pandas as pd
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
    parser.add_argument(
        "--train-parquet-path", default=None,
        help="train on this parquet WHOLESALE (all its images; bypasses the "
             "canonical split, which misbehaves on small parquets) while the "
             "val loader keeps --parquet-path's canonical val. For generated "
             "pretraining data (e.g. ScanDiff-generated scanpaths).")
    parser.add_argument(
        "--train-image-subset-parquet", default=None,
        help="restrict TRAINING images to those present in this parquet "
             "(data still comes from --parquet-path). For image-paired "
             "controls against a --train-parquet-path run.")
    parser.add_argument(
        "--train-image-list", default=None,
        help="text file with one image per line, as 'class/file.JPEG' or a full "
             "path; only these images are used for training. Matching is on the "
             "last two path components, so the protocol split files can be used "
             "directly. Overrides --max-images, which selects images in parquet "
             "order and therefore -- since the parquet is sorted by class -- "
             "gives degenerate class coverage at small scales.")
    parser.add_argument(
        "--train-scanpath-ids", type=int, nargs="+", default=None,
        help="keep only these scanpath (epoch) IDs for training, e.g. 0 1 2 3 4. "
             "Without this the sampler draws a fresh subset of the 16 available "
             "paths every epoch, so the model eventually sees all of them; "
             "DeepGaze III and ScanDiff both train on a fixed 0-4.")
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
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    # training loss
    parser.add_argument(
        "--loss", choices=["kl", "ll"], default="kl",
        help="training objective: 'kl' = KL divergence against Gaussian heatmap "
             "(requires --heatmap-sigma); 'll' = NLL at the true fixation location "
             "(no sigma needed, sharper signal)",
    )

    # model
    parser.add_argument("--visual-encoder", default="dinov2_base")
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--decoder-dropout", type=float, default=None,
        help="separate dropout rate for the decoder's Dropout2d layers. "
             "if unset, inherits --dropout. set to 0.0 to disable decoder dropout "
             "while keeping it elsewhere (recommended — Dropout2d zeroes entire "
             "feature channels, which is disruptive for spatial heatmap structure).",
    )
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)
    parser.add_argument("--decoder-upsample", default="nearest",
                        choices=["nearest", "bilinear", "transposed"],
                        help="upsampling used in the heatmap decoder. 'nearest' is "
                             "the default the whole ladder was trained with; the "
                             "other two exist for the decoder ablation.")
    parser.add_argument("--max-seq-len", type=int, default=16,
                        help="maximum total scanpath length (full sequence including first fixation)")
    parser.add_argument(
        "--visual-scanpath-features", action="store_true",
        help="enrich each fixation token with the DINOv2 patch feature at that location, "
             "so the scanpath encoder reasons over the visual sequence not just coordinates",
    )
    parser.add_argument(
        "--patch-pos-embed", action="store_true",
        help="add learned 2D patch position embeddings to image tokens after the adapter. "
             "addresses coordinate-space mismatch between frozen DINOv2 PEs and the "
             "scanpath encoder's coord_proj space; trained end-to-end with our objective.",
    )
    parser.add_argument(
        "--bidirectional-fusion", action="store_true",
        help="use parallel co-attention in fusion layers: image patches and scanpath tokens "
             "mutually attend to each other (both directions computed in parallel using "
             "pre-update representations as K, V). doubles fusion layer parameter count.",
    )
    parser.add_argument(
        "--fixation-query-fusion", action="store_true",
        help="reverse fusion direction: fixation tokens attend to image patches (instead of "
             "image patches attending to fixation tokens). produces enriched fixation memories "
             "which are then decoded via a second cross-attention. mutually exclusive with "
             "--bidirectional-fusion.",
    )
    parser.add_argument(
        "--use-finalizer", action="store_true",
        help="DG3-style readout finalizer: learnable Gaussian blur on the predicted "
             "heatmap before log-softmax (optimal LL smoothing under fixation noise).",
    )
    parser.add_argument(
        "--use-centerbias", action="store_true",
        help="add a DG3-style additive center-bias log-prior (learnable weight) to the "
             "readout. For --dataset mit1003 the prior is fit on the training fold; "
             "implies a finalizer. No-op unless a center bias is fit.",
    )
    parser.add_argument(
        "--use-saccadeprior", action="store_true",
        help="add a DG3-inspired additive saccade-delta log-prior (learnable weight "
             "sacc_alpha) to the readout: a 1st-order Markov p(next_cell - prev_cell) "
             "kernel, fit on the training fold and placed at each step's previous-"
             "fixation cell. For --dataset mit1003 only; implies a finalizer.",
    )
    parser.add_argument(
        "--finalizer-init-sigma", type=float, default=1.0,
        help="initial Gaussian blur sigma (in heatmap cells) for --use-finalizer.",
    )

    # optimization
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    # fine-tuning
    parser.add_argument("--dataset", choices=["synthetic", "mit1003"], default="synthetic",
                        help="'synthetic' = ImageNet parquet scanpaths; 'mit1003' = human fixations")
    parser.add_argument("--init-from", default=None,
                        help="load model weights only (fresh optimizer/schedule) for fine-tuning. "
                             "Unlike --resume, does not restore optimizer/epoch/step.")
    parser.add_argument("--freeze", nargs="*", default=[],
                        help="module name prefixes to freeze, e.g. scanpath_encoder fusion image_adapter")
    parser.add_argument("--readout-only", action="store_true",
                        help="freeze everything except the decoder (spatial readout head)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="fraction of MIT1003 images held out for validation (image-wise split)")
    parser.add_argument("--cv-fold", type=int, default=None,
                        help="MIT1003 cross-validation: which fold (0-indexed) to use as validation. "
                             "Overrides --val-frac when set.")
    parser.add_argument("--cv-num-folds", type=int, default=10,
                        help="number of cross-validation folds (only used with --cv-fold)")
    parser.add_argument("--mit-split-dir", type=str, default=None,
                        help="MIT1003 FIXED splits: directory with train.txt / validation.txt "
                             "(one image filename per line), e.g. the authors' ScanDiff splits "
                             "in thesis_protocol/v2/splits/mit1003. Replaces --cv-fold / "
                             "--val-frac; the leakage-proof protocol for the fine-tuning wave "
                             "(the official SDgen teacher trained on the authors' train split, "
                             "so CV folds over all 1003 images are contaminated). test.txt is "
                             "never read here.")
    parser.add_argument("--no-save-checkpoints", action="store_true",
                        help="skip writing model checkpoints (only log.csv). For CV folds / "
                             "sweeps where the trained weights are disposable — saves disk.")

    # wandb
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fixation-prediction")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None,
                        help="wandb group — e.g. set to the CV condition so all 10 folds "
                             "appear as one group with automatic mean/std across folds")
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
        group=args.wandb_group,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _denormalise_image(image_chw: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3, H, W) torch image to a [0, 1] HxWx3 numpy array."""
    img = image_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def log_wandb_samples(
    run,
    image: torch.Tensor,            # (B, 3, H_img, W_img)
    fixations: torch.Tensor,        # (B, max_seq_len, 2) in [-1, 1]
    fixations_len: torch.Tensor,    # (B,) int
    log_preds: torch.Tensor,        # (B, T, H, W)
    step: int,
    n_samples: int = 4,
    tag: str = "train",
) -> None:
    """
    Rich per-sample visualisation: input image, past fixations as a connected
    path, target fixation as a star, predicted heatmap overlaid. For each
    sample we show three prediction steps (early/middle/late within the valid
    prefix range) so you can see how the prediction evolves with history.

    Logged as a single wandb.Image (matplotlib figure) at key f"{tag}/samples".
    Wrapped in try/except so a viz failure can't kill training.
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import wandb

    try:
        B, _, img_h, img_w = image.shape
        T = log_preds.shape[1]
        n_samples = min(n_samples, B)

        # find samples with a long-enough prefix for a meaningful 3-step view.
        # prefix_len_b = fixations_len_b - 1 is the number of valid prediction steps.
        prefix_lens = (fixations_len - 1).clamp(min=0).cpu().numpy()
        candidates = [b for b in range(B) if prefix_lens[b] >= 2][:n_samples]
        if not candidates:
            return

        # upsample all log_preds for the chosen samples to image resolution once
        # (much cheaper than per-step PIL ops).
        chosen_logp = log_preds[candidates].detach().float()  # (k, T, H, W)
        chosen_pred = chosen_logp.exp()
        chosen_pred_up = F.interpolate(
            chosen_pred,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).cpu().numpy()                              # (k, T, img_h, img_w)

        n = len(candidates)
        fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), squeeze=False)

        for row, b in enumerate(candidates):
            prefix_len = int(prefix_lens[b])
            steps_to_show = sorted({
                0,
                max(1, prefix_len // 2),
                prefix_len - 1,
            })
            # pad to exactly 3 columns by repeating the last step
            while len(steps_to_show) < 3:
                steps_to_show.append(steps_to_show[-1])

            img_np = _denormalise_image(image[b])  # (H, W, 3) in [0, 1]
            fix_np = fixations[b].detach().cpu().float().numpy()  # (max_seq_len, 2)
            # convert normalised coords [-1, 1] to image pixels [0, img_w/h - 1]
            fix_px_x = (fix_np[:, 0] + 1) / 2 * (img_w - 1)
            fix_px_y = (fix_np[:, 1] + 1) / 2 * (img_h - 1)

            for col, t in enumerate(steps_to_show):
                ax = axes[row, col]
                ax.imshow(img_np)
                # overlay prediction heatmap (alpha so image still visible).
                # vmax pinned per-step to avoid one peak dominating the colour map.
                pred_up = chosen_pred_up[row, t]
                ax.imshow(pred_up, cmap="hot", alpha=0.55,
                          vmin=0.0, vmax=float(pred_up.max() + 1e-12))
                # past fixations: connected blue path, growing with t.
                past_x = fix_px_x[:t + 1]
                past_y = fix_px_y[:t + 1]
                ax.plot(past_x, past_y, "-o", color="dodgerblue",
                        markersize=4, linewidth=1.2, alpha=0.9)
                # most recent fixation = larger marker.
                ax.plot(past_x[-1], past_y[-1], "o", color="dodgerblue",
                        markersize=8, markeredgecolor="white", markeredgewidth=1.2)
                # target fixation (what we're trying to predict): green star.
                ax.plot(fix_px_x[t + 1], fix_px_y[t + 1], "*", color="lime",
                        markersize=15, markeredgecolor="black", markeredgewidth=0.8)
                ax.set_title(f"sample {b}  step t={t}  (prefix={t+1})", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        run.log({f"{tag}/samples": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        # never let visualisation kill training
        print(f"warning: log_wandb_samples failed: {e}", flush=True)


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
) -> dict[str, Any]:
    model.eval()
    device_type = device.split(":")[0]

    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0

    # stash the first batch's data + predictions so the caller can visualise it.
    sample_batch: dict[str, torch.Tensor] | None = None

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

        if sample_batch is None:
            # detach + keep on GPU; log_wandb_samples handles transfer.
            sample_batch = {
                "image": image.detach(),
                "fixations": fixations.detach(),
                "fixations_len": fixations_len.detach(),
                "log_preds": log_preds.detach(),
            }

        # valid_mask[b, t] = True when step t has a real prediction target
        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < (fixations_len - 1).unsqueeze(1)
        )  # (B, T)

        # flatten valid (sample, step) pairs for per-step metrics
        valid_flat = valid_mask.view(-1)
        if not valid_flat.any():
            continue

        lp = log_preds.reshape(B * T, *log_preds.shape[2:])[valid_flat]       # (V, H, W)
        th = target_heatmaps.reshape(B * T, *target_heatmaps.shape[2:])[valid_flat]  # (V, H, W)
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
                "val_nss": float("nan"), "val_auc": float("nan"),
                "val_steps": 0, "sample_batch": None}

    return {
        "val_kl":   total_kl   / total_valid,
        "val_ll":   total_ll   / total_valid,
        "val_nss":  total_nss  / total_valid,
        "val_auc":  total_auc  / total_valid,
        "val_steps": total_valid,
        "sample_batch": sample_batch,
    }


def run_validation(
    model, val_loader, optimizer, scheduler, scaler,
    device, step, epoch, epoch_float, use_amp, args,
    log_path, checkpoint_dir, wandb_run, best_val_kl, best_val_nss,
    best_val_ll=float("-inf"),
) -> tuple[float, float, float]:
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
        sample = metrics.get("sample_batch")
        if sample is not None:
            log_wandb_samples(
                wandb_run,
                image=sample["image"],
                fixations=sample["fixations"],
                fixations_len=sample["fixations_len"],
                log_preds=sample["log_preds"],
                step=step,
                n_samples=4,
                tag="val",
            )

    if metrics["val_kl"] < best_val_kl:
        best_val_kl = metrics["val_kl"]
        if not args.no_save_checkpoints:
            best_path = checkpoint_dir / "best_val_kl.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, step, epoch, args,
                            scaler, best_val_kl, best_val_nss)
            print(f"saved new best val kl checkpoint (val_kl={best_val_kl:.4f})", flush=True)

    if metrics["val_nss"] > best_val_nss:
        best_val_nss = metrics["val_nss"]
        if not args.no_save_checkpoints:
            best_path = checkpoint_dir / "best_val_nss.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, step, epoch, args,
                            scaler, best_val_kl, best_val_nss)
            print(f"saved new best val nss checkpoint (val_nss={best_val_nss:.4f})", flush=True)

    if metrics["val_ll"] > best_val_ll:
        best_val_ll = metrics["val_ll"]
        if not args.no_save_checkpoints:
            best_path = checkpoint_dir / "best_val_ll.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, step, epoch, args,
                            scaler, best_val_kl, best_val_nss)
            print(f"saved new best val ll checkpoint (val_ll={best_val_ll:.4f})", flush=True)

    return best_val_kl, best_val_nss, best_val_ll


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # benchmark=True lets cuDNN pick the fastest conv algorithm per input shape
    # deterministic=False because AMP isn't bit-exact reproducible anyway, and
    # the deterministic conv path is significantly slower on A100
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device.split(":")[0]

    use_amp = args.amp and device == "cuda"
    if args.amp and not use_amp:
        print("warning: --amp requested but device is not cuda, disabling", flush=True)

    # The image transform is chosen at import time in src/data/dataset.py from an
    # environment variable, so it is invisible in the checkpoint unless recorded
    # here. Without this, a crop-trained run cannot be told apart from a
    # resize-trained one after the fact, and evaluating one with the other's
    # transform silently produces a train/eval mismatch.
    args.ours_crop_transform = os.environ.get("OURS_CROP_TRANSFORM") == "1"
    # ImageGroupedSampler guarantees consecutive same-image groups only for
    # the synthetic parquet dataset. MIT1003 uses a regular shuffled loader;
    # reusing one image encoding for `paths_per_image` consecutive records
    # there would silently assign features from the wrong images.
    args.effective_paths_per_image = (
        args.paths_per_image if args.dataset == "synthetic" else 1
    )
    if args.dataset != "synthetic" and args.paths_per_image != 1:
        print(
            f"forcing effective_paths_per_image=1 for {args.dataset} "
            f"(requested/default value was {args.paths_per_image})",
            flush=True,
        )

    print(f"device: {device}", flush=True)
    print(f"amp: {use_amp}", flush=True)
    print(f"image transform: {'Resize(256)+CenterCrop(224)' if args.ours_crop_transform else 'Resize((224,224))'}", flush=True)
    print("args:", vars(args), flush=True)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "config.json", vars(args))

    wandb_run = init_wandb(args)

    if args.dataset == "mit1003":
        from src.data.mit1003_dataset import make_mit1003_loader, MIT1003Dataset

        # image-wise train/val split: val images are completely unseen.
        n_images = 1003
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_images)

        if args.mit_split_dir is not None:
            # Fixed authors' splits (leakage-proof fine-tuning protocol).
            assert args.cv_fold is None, "--mit-split-dir and --cv-fold are mutually exclusive"
            import h5py
            from src.data.mit1003_dataset import MIT1003_ROOT
            split_dir = Path(args.mit_split_dir)
            with h5py.File(MIT1003_ROOT / "stimuli.hdf5", "r") as f:
                fnames = [fn.decode() if isinstance(fn, bytes) else fn
                          for fn in f["filenames"][:]]
            name_to_idx = {Path(fn).name.lower(): i for i, fn in enumerate(fnames)}

            def _load_split(fname):
                wanted = [l.strip() for l in open(split_dir / fname) if l.strip()]
                missing = [w for w in wanted if Path(w).name.lower() not in name_to_idx]
                if missing:
                    raise SystemExit(f"{fname}: {len(missing)} images not found in "
                                     f"stimuli.hdf5, e.g. {missing[:3]}")
                return sorted(name_to_idx[Path(w).name.lower()] for w in wanted)

            train_images = _load_split("train.txt")
            val_images = _load_split("validation.txt")
            overlap = set(train_images) & set(val_images)
            assert not overlap, f"train/val overlap: {len(overlap)} images"
            print(f"MIT1003 FIXED splits from {split_dir}: "
                  f"{len(train_images)} train, {len(val_images)} val images", flush=True)
        elif args.cv_fold is not None:
            # k-fold CV: deterministically partition the shuffled images into
            # cv_num_folds contiguous chunks; fold cv_fold is the validation set.
            # Every image is a validation image in exactly one fold.
            folds = np.array_split(perm, args.cv_num_folds)
            val_images   = sorted(folds[args.cv_fold].tolist())
            train_images = sorted(np.concatenate(
                [folds[i] for i in range(args.cv_num_folds) if i != args.cv_fold]
            ).tolist())
            print(f"MIT1003 CV fold {args.cv_fold}/{args.cv_num_folds}: "
                  f"{len(train_images)} train, {len(val_images)} val images", flush=True)
        else:
            n_val = int(round(args.val_frac * n_images))
            val_images   = sorted(perm[:n_val].tolist())
            train_images = sorted(perm[n_val:].tolist())
            print(f"MIT1003 split: {len(train_images)} train, {len(val_images)} val images", flush=True)

        train_loader = make_mit1003_loader(
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_seq_len=args.max_seq_len, image_indices=train_images,
            shuffle=True, seed=args.seed,
        )
        val_loader = make_mit1003_loader(
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_seq_len=args.max_seq_len, image_indices=val_images,
            shuffle=False, seed=args.seed,
        )
    else:
        train_parquet = args.train_parquet_path or args.parquet_path
        train_subset = None
        if args.train_image_list:
            # protocol split files store 'class/file.JPEG'; the parquet stores a
            # longer prefixed path. Match on the last two components, and fail
            # loudly rather than silently training on a subset of the subset.
            def _key(p):
                a = p.strip().split("/")
                return a[-2] + "/" + a[-1]

            wanted = {_key(l) for l in open(args.train_image_list) if l.strip()}
            available = (pd.read_parquet(train_parquet, engine="pyarrow",
                                         columns=["image_path"])["image_path"]
                         .unique().tolist())
            by_key = {_key(p): p for p in available}
            missing = wanted - by_key.keys()
            if missing:
                raise SystemExit(
                    f"{len(missing)} of {len(wanted)} images from "
                    f"{args.train_image_list} are not in {train_parquet}, "
                    f"e.g. {sorted(missing)[:3]}")
            train_subset = [by_key[k] for k in sorted(wanted)]
            print(f"training restricted to {len(train_subset)} images "
                  f"from {args.train_image_list}", flush=True)
        elif args.train_parquet_path or args.train_image_subset_parquet:
            subset_src = args.train_parquet_path or args.train_image_subset_parquet
            train_subset = (
                pd.read_parquet(subset_src, engine="pyarrow", columns=["image_path"])
                ["image_path"].unique().tolist()
            )
            print(f"training restricted to {len(train_subset)} images "
                  f"from {subset_src}", flush=True)

        if args.train_scanpath_ids:
            print(f"training scanpath IDs fixed to {args.train_scanpath_ids} "
                  f"(no per-epoch resampling)", flush=True)

        train_loader = make_dataloader(
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            max_images=(None if train_subset else args.max_images),
            use_grouped_sampler=True,
            paths_per_image=args.paths_per_image,
            max_seq_len=args.max_seq_len,
            parquet_path=train_parquet,
            image_subset=train_subset,
            epoch_subset=args.train_scanpath_ids,
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
        decoder_dropout=args.decoder_dropout,
        decoder_hidden_channels=args.decoder_hidden_channels,
        decoder_upsample=args.decoder_upsample,
        use_visual_scanpath_features=args.visual_scanpath_features,
        use_patch_pos_embed=args.patch_pos_embed,
        use_bidirectional_fusion=args.bidirectional_fusion,
        use_fixation_query_fusion=args.fixation_query_fusion,
        use_finalizer=args.use_finalizer,
        use_centerbias=args.use_centerbias,
        use_saccadeprior=args.use_saccadeprior,
        finalizer_init_sigma=args.finalizer_init_sigma,
    ).to(device)

    # --- fine-tuning: load pretrained weights only (fresh optimizer/schedule) ---
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location=device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"initialized from {args.init_from} | "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"  missing keys (newly-initialised): {missing[:8]}{' …' if len(missing) > 8 else ''}", flush=True)

    # --- center bias: fit on the training-fold fixations, populate the finalizer prior ---
    if args.use_centerbias:
        if args.dataset != "mit1003":
            raise ValueError("--use-centerbias currently only wired for --dataset mit1003")
        from src.eval.fair_metrics import fit_centerbias_tuned
        # all real fixations per image (synthetic centre fixation excluded,
        # untruncated), from the raw records — matches eval_mit1003_fair.py
        cb_ds = MIT1003Dataset(max_seq_len=args.max_seq_len, min_fixations=3,
                               image_indices=train_images)
        by_img: dict[int, list[np.ndarray]] = {}
        for rec in cb_ds._records:
            H, W = cb_ds._img_shapes[rec["img_idx"]]
            n = rec["n_valid"]
            x = np.clip((rec["xs"][:n] / (W - 1)) * 2.0 - 1.0, -1.0, 1.0)
            y = np.clip((rec["ys"][:n] / (H - 1)) * 2.0 - 1.0, -1.0, 1.0)
            by_img.setdefault(rec["img_idx"], []).append(
                np.stack([x, y], axis=1).astype(np.float32))
        cb_per_img = [np.concatenate(v, axis=0) for v in by_img.values()]
        cb_log = fit_centerbias_tuned(cb_per_img, args.heatmap_size,
                                      device=str(device), verbose=True)
        model.finalizer.set_centerbias(cb_log)
        n_cb = sum(len(a) for a in cb_per_img)
        print(f"fit tuned center bias on {n_cb} training fixations "
              f"({args.heatmap_size}x{args.heatmap_size} grid)", flush=True)

    # --- saccade-delta prior: 1st-order Markov p(next_cell - prev_cell) ---
    if args.use_saccadeprior:
        if args.dataset != "mit1003":
            raise ValueError("--use-saccadeprior currently only wired for --dataset mit1003")
        from src.eval.fair_metrics import fit_saccade_prior_tuned, scanpath_deltas_by_image
        # per-image list of per-SCANPATH arrays (boundaries preserved, unlike
        # the center-bias by_img above) -- deltas must never cross subjects.
        sacc_ds = MIT1003Dataset(max_seq_len=args.max_seq_len, min_fixations=3,
                                 image_indices=train_images)
        sacc_by_img: dict[int, list[np.ndarray]] = {}
        for rec in sacc_ds._records:
            H, W = sacc_ds._img_shapes[rec["img_idx"]]
            n = rec["n_valid"]
            x = np.clip((rec["xs"][:n] / (W - 1)) * 2.0 - 1.0, -1.0, 1.0)
            y = np.clip((rec["ys"][:n] / (H - 1)) * 2.0 - 1.0, -1.0, 1.0)
            sacc_by_img.setdefault(rec["img_idx"], []).append(
                np.stack([x, y], axis=1).astype(np.float32))
        sacc_per_img = [sacc_by_img[k] for k in sorted(sacc_by_img)]
        delta_per_img = scanpath_deltas_by_image(sacc_per_img)
        sacc_log = fit_saccade_prior_tuned(delta_per_img, args.heatmap_size, verbose=True)
        model.set_saccadeprior(sacc_log.to(device))
        n_delta = sum(len(a) for a in delta_per_img)
        print(f"fit tuned saccade-delta prior on {n_delta} training saccades "
              f"({args.heatmap_size}x{args.heatmap_size} grid)", flush=True)

    # --- freezing: --readout-only freezes all but the decoder; --freeze takes prefixes ---
    freeze_prefixes = list(args.freeze)
    if args.readout_only:
        # freeze everything except the SpatialHeatmapDecoder ('decoder.'),
        # i.e. keep decoder_cross_attn and all upstream modules frozen.
        for name, p in model.named_parameters():
            if not name.startswith("decoder."):
                p.requires_grad = False
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in freeze_prefixes):
            p.requires_grad = False

    # the visual encoder is frozen, so only non-encoder params are trainable.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"trainable parameters: {n_trainable:,} "
          f"({len(trainable_params)} tensors)", flush=True)

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
    best_val_ll = float("-inf")

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

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device_type, enabled=use_amp):
                    log_preds = model(
                        image,
                        fixations,
                        fixations_len,
                        args.ignore_prefix,
                        paths_per_image=args.effective_paths_per_image,
                    )

                log_preds = log_preds.float()

                if args.loss == "kl":
                    target_heatmaps = make_gaussian_heatmaps_batch(
                        target_fixations, args.heatmap_size, args.heatmap_sigma
                    ).float()
                    loss_kl = multi_step_kl_loss(log_preds, target_heatmaps, fixations_len)
                    loss_ll = multi_step_ll_score(log_preds, target_fixations, fixations_len)
                    training_loss = loss_kl
                else:
                    target_heatmaps = make_gaussian_heatmaps_batch(
                        target_fixations, args.heatmap_size, args.heatmap_sigma
                    ).float()
                    loss_kl = multi_step_kl_loss(log_preds, target_heatmaps, fixations_len)
                    loss_ll = multi_step_ll_score(log_preds, target_fixations, fixations_len)
                    training_loss = -loss_ll

                if not torch.isfinite(training_loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {training_loss.item()}")

                scaler.scale(training_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.grad_clip, error_if_nonfinite=False,
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
                        modules_to_log = [
                            ("scanpath_encoder", model.scanpath_encoder),
                            ("fusion", model.fusion),
                            ("decoder", model.decoder),
                            ("image_adapter", model.image_adapter),
                        ]
                        if getattr(model, "use_patch_pos_embed", False):
                            modules_to_log.append(("patch_pos_embed", model.patch_pos_embed))
                        for name, module in modules_to_log:
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
                    log_wandb_samples(
                        wandb_run,
                        image=image,
                        fixations=fixations,
                        fixations_len=fixations_len,
                        log_preds=log_preds,
                        step=step,
                        n_samples=4,
                        tag="train",
                    )

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

            best_val_kl, best_val_nss, best_val_ll = run_validation(
                model=model, val_loader=val_loader, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, device=device,
                step=step, epoch=epoch, epoch_float=float(epoch + 1),
                use_amp=use_amp, args=args, log_path=log_path,
                checkpoint_dir=checkpoint_dir, wandb_run=wandb_run,
                best_val_kl=best_val_kl, best_val_nss=best_val_nss,
                best_val_ll=best_val_ll,
            )

            if not args.no_save_checkpoints:
                latest_path = checkpoint_dir / "latest.pt"
                save_checkpoint(latest_path, model, optimizer, scheduler,
                                step, epoch, args, scaler, best_val_kl, best_val_nss)
                print(f"saved latest checkpoint: {latest_path}", flush=True)

        if not args.no_save_checkpoints:
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
