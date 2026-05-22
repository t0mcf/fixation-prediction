"""
Qualitative visualization of scanpath model predictions.

Each panel shows:
  - image in the background
  - predicted heatmap overlaid (hot colormap, semi-transparent)
  - scanpath history as numbered circles (1 = first fixation, 2 = second, ...)
  - ground truth next fixation as a green star

Four plot types:
  1. evolution    — all prediction steps for a single sample (how history grows)
  2. multi_model  — same image/step across multiple checkpoints side by side
  3. good_cases   — steps where the model assigns highest LL to the true fixation
  4. bad_cases    — steps where the model assigns lowest LL to the true fixation

Usage (single model):
    python scripts/visualize_predictions.py \\
        --checkpoints runs/train_200k_ll_15ep/checkpoints/best_val_nss.pt \\
        --output-dir runs/viz/base

Usage (multi-model comparison):
    python scripts/visualize_predictions.py \\
        --checkpoints runs/train_200k_ll_15ep/checkpoints/best_val_nss.pt \\
                      runs/train_200k_ll_patchpos_15ep/checkpoints/best_val_nss.pt \\
        --model-names base patchpos \\
        --output-dir runs/viz/comparison
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
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

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def denorm_image(image_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) numpy array in [0, 1]."""
    img = image_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)


def to_pixels(xy_norm: np.ndarray, H: int, W: int) -> tuple[float, float]:
    """[-1, 1] normalised coordinate → pixel position (px, py)."""
    px = (xy_norm[0] + 1.0) / 2.0 * (W - 1)
    py = (xy_norm[1] + 1.0) / 2.0 * (H - 1)
    return float(px), float(py)


def compute_ll_per_step(
    log_pred_step: torch.Tensor,   # (H, W) log-softmax
    gt_xy: np.ndarray,             # (2,) normalised
) -> float:
    H, W = log_pred_step.shape
    col = round((gt_xy[0] + 1.0) / 2.0 * (W - 1))
    row = round((gt_xy[1] + 1.0) / 2.0 * (H - 1))
    col = max(0, min(W - 1, col))
    row = max(0, min(H - 1, row))
    return float(log_pred_step[row, col].item())


def build_model(checkpoint: dict[str, Any], device: str) -> ScanpathModel:
    a = checkpoint.get("args", {})
    model = ScanpathModel(
        visual_encoder=a.get("visual_encoder", "dinov2_base"),
        pretrained_encoder=not a.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=a.get("d_model", 256),
        n_heads=a.get("n_heads", 8),
        scanpath_layers=a.get("scanpath_layers", 2),
        fusion_layers=a.get("fusion_layers", 1),
        max_seq_len=a.get("max_seq_len", 16),
        heatmap_size=a.get("heatmap_size", 64),
        dropout=a.get("dropout", 0.1),
        decoder_dropout=a.get("decoder_dropout", None),
        decoder_hidden_channels=a.get("decoder_hidden_channels", 256),
        use_visual_scanpath_features=a.get("visual_scanpath_features", False),
        use_patch_pos_embed=a.get("patch_pos_embed", False),
        use_bidirectional_fusion=a.get("bidirectional_fusion", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# core drawing function
# ---------------------------------------------------------------------------

def draw_panel(
    ax: plt.Axes,
    image_np: np.ndarray,         # (H, W, 3) in [0, 1]
    log_pred: torch.Tensor,       # (H_map, W_map) log-softmax
    fixations_np: np.ndarray,     # (seq_len, 2) normalised
    step_t: int,                  # prefix = fixations[0..step_t], predict fixations[step_t+1]
    fixation_len: int,
    ll_val: float | None = None,
    title: str | None = None,
) -> None:
    H_img, W_img = image_np.shape[:2]

    # background image
    ax.imshow(image_np)

    # heatmap overlay: upsample to image resolution
    pred = log_pred.exp().detach().cpu().float()
    pred_up = F.interpolate(
        pred[None, None],
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    vmax = float(pred_up.max()) + 1e-12
    ax.imshow(pred_up, cmap="hot", alpha=0.5, vmin=0.0, vmax=vmax)

    # numbered history fixations
    for i in range(step_t + 1):
        if i >= fixation_len:
            break
        px, py = to_pixels(fixations_np[i], H_img, W_img)
        ax.scatter(px, py, s=90, color="dodgerblue", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(
            px, py, str(i + 1),
            color="white", fontsize=6, ha="center", va="center",
            fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
        )

    # ground truth next fixation: green star
    next_idx = step_t + 1
    if next_idx < fixation_len:
        gt_px, gt_py = to_pixels(fixations_np[next_idx], H_img, W_img)
        ax.plot(gt_px, gt_py, "*", color="lime", markersize=13,
                markeredgecolor="black", markeredgewidth=0.7, zorder=7)

    # title
    if title is None:
        title = f"step t={step_t}  (prefix {step_t + 1})"
    if ll_val is not None:
        title += f"\nLL={ll_val:.3f}"
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# plot type 1: evolution
# ---------------------------------------------------------------------------

def plot_evolution(
    samples: list[dict],    # list of {image, fixations, fixations_len, log_preds}
    output_dir: Path,
    num_samples: int = 4,
) -> None:
    """One figure per sample: columns = prediction steps (t=0, t=1, ..., t=N-2)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = samples[:num_samples]

    for s_idx, s in enumerate(chosen):
        img_np      = denorm_image(s["image"])
        fix_np      = s["fixations"].cpu().numpy()        # (S, 2)
        fix_len     = int(s["fixations_len"])
        log_preds   = s["log_preds"]                      # (T, H, W)
        T           = log_preds.shape[0]
        n_valid     = max(1, fix_len - 1)
        steps       = list(range(min(n_valid, T)))

        ncols = len(steps)
        fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]
        fig.suptitle(f"Evolution — sample {s_idx}", fontsize=9)

        for col, t in enumerate(steps):
            ll_val = compute_ll_per_step(log_preds[t], fix_np[t + 1])
            draw_panel(
                axes[col], img_np, log_preds[t],
                fix_np, t, fix_len, ll_val=ll_val,
            )

        plt.tight_layout()
        out_path = output_dir / f"evolution_sample{s_idx:02d}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# plot type 2: multi-model comparison
# ---------------------------------------------------------------------------

def plot_multi_model(
    samples_per_model: list[list[dict]],   # [model_idx][sample_idx]
    model_names: list[str],
    output_dir: Path,
    num_samples: int = 4,
    steps_per_sample: int = 3,
) -> None:
    """
    Rows = samples × steps, columns = models.
    For each sample, pick `steps_per_sample` evenly spaced steps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(model_names)
    chosen_idx = list(range(min(num_samples, len(samples_per_model[0]))))

    for s_idx in chosen_idx:
        # pick steps from the first model's valid range (all models see same data)
        fix_len = int(samples_per_model[0][s_idx]["fixations_len"])
        n_valid = max(1, fix_len - 1)
        step_candidates = list(range(n_valid))
        # pick evenly spaced steps
        idxs = np.linspace(0, len(step_candidates) - 1, steps_per_sample, dtype=int)
        steps = [step_candidates[i] for i in idxs]

        nrows = len(steps)
        ncols = n_models
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(f"Multi-model comparison — sample {s_idx}", fontsize=9)

        for row, t in enumerate(steps):
            for col, (model_name, model_samples) in enumerate(
                zip(model_names, samples_per_model)
            ):
                s = model_samples[s_idx]
                img_np  = denorm_image(s["image"])
                fix_np  = s["fixations"].cpu().numpy()
                fix_len = int(s["fixations_len"])
                ll_val  = compute_ll_per_step(s["log_preds"][t], fix_np[t + 1])
                title   = model_name if row == 0 else None
                draw_panel(
                    axes[row, col], img_np, s["log_preds"][t],
                    fix_np, t, fix_len,
                    ll_val=ll_val,
                    title=title if row == 0 else f"t={t}\nLL={ll_val:.3f}",
                )

        plt.tight_layout()
        out_path = output_dir / f"comparison_sample{s_idx:02d}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# plot type 3 & 4: good / bad cases
# ---------------------------------------------------------------------------

def plot_extreme_cases(
    cases: list[dict],     # sorted list of {ll_val, image, fixations, fixations_len, log_pred_step, step_t}
    output_dir: Path,
    tag: str,              # "good" or "bad"
    n_cols: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(cases)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(f"{'Best' if tag == 'good' else 'Worst'} cases by LL", fontsize=10)

    for i, case in enumerate(cases):
        row, col = divmod(i, n_cols)
        img_np  = denorm_image(case["image"])
        fix_np  = case["fixations"].cpu().numpy()
        draw_panel(
            axes[row, col], img_np, case["log_pred_step"],
            fix_np, case["step_t"], case["fixation_len"],
            ll_val=case["ll_val"],
        )

    # hide unused axes
    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / f"{tag}_cases.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_samples(
    model: ScanpathModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int,
    scan_batches: int,
    n_good_bad: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns:
        samples      — first n_samples full sequences for evolution/comparison plots
        good_cases   — n_good_bad steps with highest LL
        bad_cases    — n_good_bad steps with lowest LL

    Memory-efficient: uses running heaps for good/bad cases so only
    2*n_good_bad case dicts are held in memory at any time regardless
    of how many batches are scanned.
    """
    samples = []

    # good_heap: min-heap of (ll_val, counter, data_dict)
    #   → we keep the n_good_bad HIGHEST ll_val entries
    #   → since it's a min-heap, the smallest-ll entry is always at the root
    #   → when full, discard root if new entry is larger
    good_heap: list = []

    # bad_heap: max-heap implemented as min-heap of (-ll_val, counter, data_dict)
    #   → we keep the n_good_bad LOWEST ll_val entries
    #   → the least-negative entry (highest ll_val among kept) is at the root
    #   → when full, discard root if new entry is more negative (smaller ll_val)
    bad_heap: list = []

    counter = 0   # unique tiebreaker so heapq never compares dicts

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= scan_batches:
            break

        image         = batch["image"].to(device)           # (B, 3, H, W)
        fixations     = batch["fixations"].to(device)       # (B, S, 2)
        fixations_len = batch["fixations_len"].to(device)   # (B,)

        B, S, _ = fixations.shape
        T = S - 1

        log_preds = model(image, fixations, fixations_len)  # (B, T, H, W)

        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < (fixations_len - 1).unsqueeze(1)
        )  # (B, T)

        # collect full samples (for evolution / multi-model)
        for b in range(B):
            if len(samples) >= n_samples:
                break
            fix_len = int(fixations_len[b].item())
            if fix_len < 3:
                continue
            samples.append({
                "image":         image[b].cpu(),
                "fixations":     fixations[b].cpu(),
                "fixations_len": fix_len,
                "log_preds":     log_preds[b].cpu(),   # (T, H, W)
            })

        # collect per-step cases for good/bad (heap-based, O(1) extra memory)
        fix_np_batch  = fixations.cpu().numpy()
        fix_len_batch = fixations_len.cpu().numpy()

        for b in range(B):
            fix_len = int(fix_len_batch[b])
            for t in range(T):
                if not valid_mask[b, t].item():
                    continue
                ll_val = compute_ll_per_step(
                    log_preds[b, t].cpu(),
                    fix_np_batch[b, t + 1],
                )
                case_data = {
                    "ll_val":        ll_val,
                    "image":         image[b].cpu(),
                    "fixations":     fixations[b].cpu(),
                    "fixation_len":  fix_len,
                    "log_pred_step": log_preds[b, t].cpu(),
                    "step_t":        t,
                }
                counter += 1

                # update good_heap (keep top n_good_bad by ll_val)
                heapq.heappush(good_heap, (ll_val, counter, case_data))
                if len(good_heap) > n_good_bad:
                    heapq.heappop(good_heap)  # evict smallest ll_val

                # update bad_heap (keep bottom n_good_bad by ll_val)
                heapq.heappush(bad_heap, (-ll_val, counter, case_data))
                if len(bad_heap) > n_good_bad:
                    heapq.heappop(bad_heap)   # evict least-negative (largest ll)

    # sort for consistent output ordering
    good_cases = [c[2] for c in sorted(good_heap, reverse=True)]
    bad_cases  = [c[2] for c in sorted(bad_heap)]

    return samples, good_cases, bad_cases


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="one or more checkpoint paths")
    parser.add_argument("--model-names", nargs="+", default=None,
                        help="display names for each checkpoint (defaults to dir name)")
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-paths-per-image", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-evolution", type=int, default=4,
                        help="samples for evolution plots")
    parser.add_argument("--num-comparison", type=int, default=4,
                        help="samples for multi-model comparison")
    parser.add_argument("--num-good-bad", type=int, default=8,
                        help="cases for good/bad plots")
    parser.add_argument("--scan-batches", type=int, default=30,
                        help="batches to scan for good/bad case selection")
    parser.add_argument("--output-dir", default="runs/viz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    output_dir = Path(args.output_dir)

    # derive model names from checkpoint paths if not given
    model_names = args.model_names
    if model_names is None:
        model_names = [Path(c).parent.parent.name for c in args.checkpoints]
    assert len(model_names) == len(args.checkpoints)

    # load all checkpoints
    checkpoints = [torch.load(c, map_location=device) for c in args.checkpoints]
    primary_ckpt = checkpoints[0]
    ckpt_args    = primary_ckpt.get("args", {})

    parquet_path  = args.parquet_path  or ckpt_args.get("parquet_path",  default_parquet_path)
    imagenet_root = args.imagenet_root or ckpt_args.get("imagenet_root", default_imagenet_root)
    max_seq_len   = ckpt_args.get("max_seq_len", 16)
    seed          = args.seed

    n_needed = max(
        args.num_evolution,
        args.num_comparison,
        args.scan_batches * args.batch_size,
    )
    max_images = args.max_images or min(5000, n_needed // 10 + 100)

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
    print(f"val samples: {len(val_loader.dataset)}", flush=True)

    # ---- collect data for each model ----
    all_model_samples = []

    for ckpt, name in zip(checkpoints, model_names):
        print(f"\ncollecting from model: {name}", flush=True)
        model = build_model(ckpt, device)
        samples, good_cases, bad_cases = collect_samples(
            model=model,
            loader=val_loader,
            device=device,
            n_samples=max(args.num_evolution, args.num_comparison),
            scan_batches=args.scan_batches,
            n_good_bad=args.num_good_bad,
        )
        all_model_samples.append(samples)

        model_out_dir = output_dir / name

        # plot 1: evolution
        print(f"  plotting evolution...", flush=True)
        plot_evolution(
            samples=samples,
            output_dir=model_out_dir / "evolution",
            num_samples=args.num_evolution,
        )

        # plot 3/4: good and bad cases
        print(f"  plotting good/bad cases...", flush=True)
        plot_extreme_cases(good_cases, model_out_dir / "good_bad", tag="good",
                           n_cols=min(4, args.num_good_bad))
        plot_extreme_cases(bad_cases,  model_out_dir / "good_bad", tag="bad",
                           n_cols=min(4, args.num_good_bad))

        del model
        torch.cuda.empty_cache()

    # plot 2: multi-model comparison (only if multiple checkpoints)
    if len(checkpoints) > 1:
        print("\nplotting multi-model comparison...", flush=True)
        plot_multi_model(
            samples_per_model=all_model_samples,
            model_names=model_names,
            output_dir=output_dir / "comparison",
            num_samples=args.num_comparison,
        )

    print(f"\ndone. outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
