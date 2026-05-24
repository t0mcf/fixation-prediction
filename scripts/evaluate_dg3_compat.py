"""
DG3-compatible evaluation script.

Evaluates a trained checkpoint using DeepGaze III's exact metric formulations
so results can be directly compared. Key differences from evaluate_checkpoint.py:

  LL  : (mean log_density_at_fixation + log(H*W)) / log(2)  [bits, uniform=0]
  NSS : identical formula, mask-based
  AUC : identical formula, mask-based (pysaliency not required for single fixation)

Resolution: our 64x64 predictions are resized to DG3's output resolution
(224x224 — scanpather imagenet.yaml uses downsample=1, saliency_map_factor=1,
and DG3's Finalizer outputs at full image resolution) so the log(H*W)
normalization term is identical on both sides. Resizing pools probability
mass (exp → bilinear interp → renorm → log).

With --dg3-val-split the checkpoint is evaluated on the *exact* validation
images scanpather (DG3) was scored on, reproduced from its split_images()
logic. Both the full 5000-image set and the contamination-free subset
(val images not in this checkpoint's own training set) are reported.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)
default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"

DG3_RESOLUTION = 224  # DG3 Finalizer upsamples back to full centerbias/image resolution


# ---------------------------------------------------------------------------
# DG3 metric functions (verbatim logic, adapted for our single-fixation setup)
# ---------------------------------------------------------------------------

def dg3_log_likelihood(log_density: torch.Tensor, fixation_mask: torch.Tensor) -> torch.Tensor:
    """
    DG3 LL: (mean log_density at fixation locations + log(H*W)) / log(2)

    input:
        log_density  : (N, H, W)  log-probability map, sums to 0 in log space
        fixation_mask: (N, H, W)  binary, exactly one 1 per sample

    output:
        scalar in bits (uniform distribution → 0, better model → positive)
    """
    H, W = log_density.shape[-2], log_density.shape[-1]
    # sum log_density * mask over spatial dims, divide by fixation count (always 1 here)
    ll = (log_density * fixation_mask).sum(dim=(-1, -2))   # (N,)
    ll = ll.mean()
    return (ll + math.log(H * W)) / math.log(2)


def dg3_nss(log_density: torch.Tensor, fixation_mask: torch.Tensor) -> torch.Tensor:
    """
    DG3 NSS: verbatim replication of DG3's nss() including the swapped
    variable names from torch.std_mean (which returns (std, mean), not (mean, std)).
    This means 'mean' below holds the std value and 'std' holds the mean value,
    exactly as in DG3's code, so results are directly comparable.

    NOTE: this is NOT a correct NSS implementation. Use only for DG3 comparison.

    input:
        log_density  : (N, H, W)
        fixation_mask: (N, H, W)  binary, exactly one 1 per sample

    output:
        scalar mean NSS (in DG3's (buggy) scale)
    """
    density = log_density.exp()
    # torch.std_mean returns (std, mean) — DG3 unpacks as (mean, std), swapping them.
    # we replicate this exactly so our numbers match theirs.
    mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std
    nss = (saliency_map * fixation_mask).sum(dim=(-1, -2))   # (N,)
    return nss.mean()


def dg3_auc(log_density: torch.Tensor, fixation_mask: torch.Tensor) -> torch.Tensor:
    """
    DG3 AUC: replicates pysaliency's auc_for_one_positive exactly.

    DG3 passes negatives = log_density.flatten() which INCLUDES the fixation
    pixel. pysaliency then computes:
        (negatives < positive).mean() + 0.5 * (negatives == positive).mean()
    where .mean() divides by len(negatives) = H*W (fixation pixel included).

    input:
        log_density  : (N, H, W)
        fixation_mask: (N, H, W)  binary, exactly one 1 per sample

    output:
        scalar mean AUC
    """
    N, H, W = log_density.shape
    flat = log_density.view(N, -1)                                    # (N, H*W)
    target_score = (log_density * fixation_mask).sum(dim=(-1, -2))   # (N,)

    # divide by H*W (fixation pixel included in negatives, matching DG3)
    lower = (flat < target_score.unsqueeze(1)).float().sum(dim=1)
    equal = (flat == target_score.unsqueeze(1)).float().sum(dim=1)   # includes fixation
    auc = (lower + 0.5 * equal) / (H * W)
    return auc.mean()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resize_log_predictions(log_preds: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resize log-probability maps from their native resolution to target_size x target_size.
    Done by pooling probability mass: exp → bilinear interp → renorm → log.

    input:  (N, H_src, W_src)  log-softmax output
    output: (N, H_tgt, W_tgt)  log-softmax at target resolution
    """
    N = log_preds.shape[0]
    probs = log_preds.exp().unsqueeze(1)                   # (N, 1, H, W)
    probs_resized = F.interpolate(
        probs, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    ).squeeze(1)                                           # (N, T, T)
    probs_resized = probs_resized.clamp_min(0)
    probs_resized = probs_resized / probs_resized.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(probs_resized.clamp_min(1e-12))


def coords_to_fixation_mask(
    xy: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Convert normalised [-1, 1] coordinates to a binary fixation mask.

    input:  xy: (N, 2)
    output: mask: (N, H, W)  with a single 1 at the nearest grid cell
    """
    N = xy.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    col = (x + 1.0) / 2.0 * (W - 1)
    row = (y + 1.0) / 2.0 * (H - 1)
    col = col.round().long().clamp(0, W - 1)
    row = row.round().long().clamp(0, H - 1)

    mask = torch.zeros(N, H, W, dtype=torch.float32, device=xy.device)
    mask[torch.arange(N, device=xy.device), row, col] = 1.0
    return mask


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> ScanpathModel:
    ckpt_args = checkpoint.get("args", {})
    model = ScanpathModel(
        visual_encoder=ckpt_args.get("visual_encoder", "dinov2_base"),
        pretrained_encoder=not ckpt_args.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=ckpt_args.get("d_model", 256),
        n_heads=ckpt_args.get("n_heads", 8),
        scanpath_layers=ckpt_args.get("scanpath_layers", 2),
        fusion_layers=ckpt_args.get("fusion_layers", 1),
        max_seq_len=ckpt_args.get("max_seq_len", 16),
        heatmap_size=ckpt_args.get("heatmap_size", 64),
        dropout=ckpt_args.get("dropout", 0.1),
        decoder_dropout=ckpt_args.get("decoder_dropout", None),
        decoder_hidden_channels=ckpt_args.get("decoder_hidden_channels", 256),
        use_visual_scanpath_features=ckpt_args.get("visual_scanpath_features", False),
        use_patch_pos_embed=ckpt_args.get("patch_pos_embed", False),
        use_bidirectional_fusion=ckpt_args.get("bidirectional_fusion", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# DG3 (scanpather) val-split reproduction
# ---------------------------------------------------------------------------

def compute_dg3_split_info(
    parquet_path: str,
    dg3_seed: int,
    dg3_n_val: int,
    ckpt_seed: int,
    ckpt_max_images: int | None,
) -> dict[str, Any]:
    """
    Reproduce scanpather's fixed validation set and our model's training set,
    so the checkpoint can be evaluated on exactly the images DG3 was scored on.

    scanpather split_images() (val_n_images branch):
        all_paths = sorted(df.image_path.unique())
        val = all_paths[rng(seed).permutation(N)[:n_val]]

    our ScanpathDataset:
        images = df.image_path.unique()              # appearance order
        rng(seed).shuffle(images)
        val   = images[:10%]
        train = (df minus val).image_path.unique()[:max_images]

    returns dict with: dg3_val, our_train, contaminated, clean (all sets of str)
    """
    col = pd.read_parquet(parquet_path, columns=["image_path"])["image_path"]

    # --- their fixed val set ---
    all_sorted = np.array(sorted(col.unique()))
    val_idx = np.random.default_rng(dg3_seed).permutation(len(all_sorted))[:dg3_n_val]
    dg3_val = set(all_sorted[val_idx].tolist())

    # --- our model's training images for this checkpoint ---
    our_train: set[str] = set()
    if ckpt_max_images:
        images = np.array(col.unique())
        np.random.default_rng(ckpt_seed).shuffle(images)
        n_val = max(1, int(len(images) * 0.1))
        our_val = set(images[:n_val].tolist())
        train_pool = col[~col.isin(our_val)]
        our_train = set(pd.unique(train_pool)[:ckpt_max_images].tolist())

    contaminated = dg3_val & our_train
    clean = dg3_val - our_train
    return {
        "dg3_val": dg3_val,
        "our_train": our_train,
        "contaminated": contaminated,
        "clean": clean,
    }


# ---------------------------------------------------------------------------
# evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ScanpathModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    eval_resolution: int,
) -> dict[str, float]:
    model.eval()

    total_ll  = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        image        = batch["image"].to(device)
        fixations    = batch["fixations"].to(device)       # (B, S, 2)
        fixations_len = batch["fixations_len"].to(device)  # (B,)

        B, S, _ = fixations.shape
        T = S - 1

        log_preds = model(image, fixations, fixations_len)  # (B, T, H, W)

        target_fixations = fixations[:, 1:, :]             # (B, T, 2)

        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < (fixations_len - 1).unsqueeze(1)
        )                                                  # (B, T)
        valid_flat = valid_mask.view(-1)
        if not valid_flat.any():
            continue

        # flatten to (N, H, W) keeping only valid steps
        lp  = log_preds.reshape(B * T, *log_preds.shape[2:])[valid_flat]   # (N, H, W)
        txy = target_fixations.reshape(B * T, 2)[valid_flat]                # (N, 2)
        n_valid = int(valid_flat.sum().item())

        # resize to DG3 resolution
        lp_resized = resize_log_predictions(lp, eval_resolution)            # (N, R, R)

        # build fixation masks at DG3 resolution
        fix_mask = coords_to_fixation_mask(txy, eval_resolution, eval_resolution)  # (N, R, R)

        total_ll  += dg3_log_likelihood(lp_resized, fix_mask).item() * n_valid
        total_nss += dg3_nss(lp_resized, fix_mask).item() * n_valid
        total_auc += dg3_auc(lp_resized, fix_mask).item() * n_valid
        total_valid += n_valid

        if batch_idx % 10 == 0:
            print(
                f"batch {batch_idx:04d} | valid={n_valid} | "
                f"ll={total_ll/total_valid:.4f} bits | "
                f"nss={total_nss/total_valid:.4f} | "
                f"auc={total_auc/total_valid:.4f}",
                flush=True,
            )

    if total_valid == 0:
        return {"steps": 0, "ll_bits": float("nan"), "nss": float("nan"), "auc": float("nan")}

    return {
        "steps":   total_valid,
        "ll_bits": total_ll  / total_valid,
        "nss":     total_nss / total_valid,
        "auc":     total_auc / total_valid,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint using DG3-compatible metrics."
    )
    parser.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-paths-per-image", type=int, default=None,
                        help="cap scanpaths per image (e.g. 10 to match DG3 eval protocol)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=-1,
                        help="number of batches to evaluate; -1 = full val set")
    parser.add_argument("--eval-resolution", type=int, default=DG3_RESOLUTION,
                        help=f"heatmap resolution for evaluation (default: {DG3_RESOLUTION}, matching DG3)")
    parser.add_argument("--dg3-val-split", action="store_true",
                        help="evaluate on the reproduced scanpather (DG3) fixed val set; "
                             "reports both the full set and the contamination-free subset")
    parser.add_argument("--dg3-seed", type=int, default=3141,
                        help="seed for scanpather's split_images (base.yaml: 3141)")
    parser.add_argument("--dg3-n-val", type=int, default=5000,
                        help="number of val images in scanpather's split (val_n_images)")
    parser.add_argument("--dg3-n-scanpaths", type=int, default=10,
                        help="scanpaths per image: keeps epochs 0..N-1 (val_n_scanpaths)")
    return parser.parse_args()


def run_dg3_val_split(
    args: argparse.Namespace,
    model: ScanpathModel,
    device: str,
    ckpt_args: dict[str, Any],
    parquet_path: str,
    imagenet_root: str,
    max_seq_len: int,
) -> None:
    """Evaluate on the reproduced scanpather fixed val set (full set + clean subset)."""
    ckpt_seed       = ckpt_args.get("seed", 42)
    ckpt_max_images = ckpt_args.get("max_images", None)

    print("\n" + "=" * 64, flush=True)
    print("reproducing scanpather (DG3) fixed val set", flush=True)
    print("=" * 64, flush=True)
    info = compute_dg3_split_info(
        parquet_path, args.dg3_seed, args.dg3_n_val, ckpt_seed, ckpt_max_images,
    )
    dg3_val      = info["dg3_val"]
    contaminated = info["contaminated"]
    clean        = info["clean"]

    print(f"DG3 val images       : {len(dg3_val)}  "
          f"(seed={args.dg3_seed}, sorted, perm[:{args.dg3_n_val}])", flush=True)
    if ckpt_max_images:
        print(f"checkpoint train set : {len(info['our_train'])} images "
              f"(seed={ckpt_seed}, max_images={ckpt_max_images})", flush=True)
        print(f"contaminated         : {len(contaminated)} "
              f"({100 * len(contaminated) / len(dg3_val):.1f}% of val also in training set)",
              flush=True)
        print(f"clean subset         : {len(clean)} images", flush=True)
    else:
        print("checkpoint has no max_images in args — cannot compute "
              "contamination; FULL == CLEAN", flush=True)
    epoch_subset = list(range(args.dg3_n_scanpaths))
    print(f"scanpaths per image  : epochs {epoch_subset[0]}..{epoch_subset[-1]}", flush=True)

    def _eval(image_set: set, tag: str) -> dict[str, float]:
        loader = make_dataloader(
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_grouped_sampler=False,
            max_seq_len=max_seq_len,
            parquet_path=parquet_path,
            imagenet_root=imagenet_root,
            image_subset=image_set,
            epoch_subset=epoch_subset,
        )
        print(f"\n[{tag}] dataset rows: {len(loader.dataset)}  — starting evaluation...",
              flush=True)
        return evaluate(model, loader, device, args.eval_batches, args.eval_resolution)

    m_full = _eval(dg3_val, "FULL")
    if ckpt_max_images and len(contaminated) > 0:
        m_clean = _eval(clean, "CLEAN")
    else:
        m_clean = m_full

    print("\n" + "=" * 64, flush=True)
    print("DG3-compatible evaluation — reproduced scanpather val set", flush=True)
    print("=" * 64, flush=True)
    print(f"{'':<20}{'FULL':>16}{'CLEAN':>16}", flush=True)
    print(f"{'images':<20}{len(dg3_val):>16}{len(clean):>16}", flush=True)
    print(f"{'valid steps':<20}{int(m_full['steps']):>16}{int(m_clean['steps']):>16}", flush=True)
    print(f"{'LL (bits)':<20}{m_full['ll_bits']:>16.4f}{m_clean['ll_bits']:>16.4f}", flush=True)
    print(f"{'NSS (DG3-buggy)':<20}{m_full['nss']:>16.4f}{m_clean['nss']:>16.4f}", flush=True)
    print(f"{'AUC':<20}{m_full['auc']:>16.4f}{m_clean['auc']:>16.4f}", flush=True)
    print("=" * 64, flush=True)
    print("CLEAN = contamination-free subset (val images not in this checkpoint's", flush=True)
    print("        training set). Use CLEAN for the honest comparison with DG3.", flush=True)
    print("=" * 64, flush=True)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}", flush=True)
    print(f"eval resolution: {args.eval_resolution}x{args.eval_resolution} "
          f"(DG3 native = {DG3_RESOLUTION}x{DG3_RESOLUTION})", flush=True)
    print(f"LL unit: bits, uniform baseline = 0.0", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args  = checkpoint.get("args", {})

    parquet_path   = args.parquet_path   or ckpt_args.get("parquet_path",   default_parquet_path)
    imagenet_root  = args.imagenet_root  or ckpt_args.get("imagenet_root",  default_imagenet_root)
    max_images     = args.max_images     or ckpt_args.get("max_images",     None)
    seed           = args.seed           or ckpt_args.get("seed",           42)
    max_seq_len    = ckpt_args.get("max_seq_len", 16)

    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"step: {checkpoint.get('step', '?')}  epoch: {checkpoint.get('epoch', '?')}", flush=True)

    model = build_model_from_checkpoint(checkpoint, device)

    if args.dg3_val_split:
        run_dg3_val_split(args, model, device, ckpt_args,
                          parquet_path, imagenet_root, max_seq_len)
        return

    # ---- default: our own seed-based 90/10 val split ----
    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        max_images=max_images,
        max_paths_per_image=args.max_paths_per_image,
        use_grouped_sampler=False,
        max_seq_len=max_seq_len,
        parquet_path=parquet_path,
        imagenet_root=imagenet_root,
    )

    print(f"\nval samples: {len(val_loader.dataset)}", flush=True)
    print("starting evaluation...\n", flush=True)

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.eval_batches,
        eval_resolution=args.eval_resolution,
    )

    print("\n" + "=" * 50, flush=True)
    print("DG3-compatible evaluation results", flush=True)
    print("=" * 50, flush=True)
    print(f"valid steps : {int(metrics['steps'])}", flush=True)
    print(f"LL (bits)   : {metrics['ll_bits']:.4f}  [uniform=0, higher=better]", flush=True)
    print(f"NSS         : {metrics['nss']:.4f}", flush=True)
    print(f"AUC         : {metrics['auc']:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
