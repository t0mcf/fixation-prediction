"""evaluate a trained scanpath model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel
from src.training.losses import make_gaussian_heatmaps_batch, kl_loss, ll_score
from src.training.metrics import nss_score, auc_score


default_parquet_path = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)

default_imagenet_root = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--imagenet-root", default=None)

    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=100)

    parser.add_argument("--heatmap-size", type=int, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=None)

    return parser.parse_args()


def get_ckpt_arg(ckpt_args: dict[str, Any], cli_value: Any, name: str, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(name, fallback)


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


@torch.no_grad()
def evaluate(
    model: ScanpathModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    heatmap_size: int,
    heatmap_sigma: float,
) -> dict[str, float]:
    model.eval()

    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        image = batch["image"].to(device)
        fixations = batch["fixations"].to(device)        # (B, max_seq_len, 2)
        fixations_len = batch["fixations_len"].to(device)  # (B,)

        B, max_seq_len, _ = fixations.shape
        T = max_seq_len - 1

        log_preds = model(image, fixations, fixations_len)  # (B, T, H, W)

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

        lp = log_preds.reshape(B * T, heatmap_size, heatmap_size)[valid_flat]
        th = target_heatmaps.reshape(B * T, heatmap_size, heatmap_size)[valid_flat]
        txy = target_fixations.reshape(B * T, 2)[valid_flat]
        n_valid = valid_flat.sum().item()

        total_kl  += kl_loss(lp, th).item() * n_valid
        total_ll  += ll_score(lp, txy).item() * n_valid
        total_nss += nss_score(lp, txy).item() * n_valid
        total_auc += auc_score(lp, txy).item() * n_valid
        total_valid += n_valid

        if batch_idx % 10 == 0:
            print(
                f"batch {batch_idx:04d} | "
                f"valid={n_valid} | "
                f"kl={total_kl/total_valid:.4f} | "
                f"nss={total_nss/total_valid:.4f} | "
                f"auc={total_auc/total_valid:.4f}",
                flush=True,
            )

    if total_valid == 0:
        return {"steps": 0, "kl": float("nan"), "ll": float("nan"),
                "nss": float("nan"), "auc": float("nan")}

    return {
        "steps": total_valid,
        "kl":  total_kl  / total_valid,
        "ll":  total_ll  / total_valid,
        "nss": total_nss / total_valid,
        "auc": total_auc / total_valid,
    }


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    print("checkpoint step:", checkpoint.get("step", "unknown"), flush=True)
    print("checkpoint epoch:", checkpoint.get("epoch", "unknown"), flush=True)

    eval_parquet_path = get_ckpt_arg(ckpt_args, args.parquet_path, "parquet_path", default_parquet_path)
    eval_imagenet_root = get_ckpt_arg(ckpt_args, args.imagenet_root, "imagenet_root", default_imagenet_root)
    eval_max_images = get_ckpt_arg(ckpt_args, args.max_images, "max_images", 1000)
    eval_seed = get_ckpt_arg(ckpt_args, args.seed, "seed", 42)
    eval_heatmap_size = get_ckpt_arg(ckpt_args, args.heatmap_size, "heatmap_size", 64)
    eval_heatmap_sigma = get_ckpt_arg(ckpt_args, args.heatmap_sigma, "heatmap_sigma", 2.0)
    eval_max_seq_len = ckpt_args.get("max_seq_len", 16)

    print("resolved eval settings:", flush=True)
    print(f"  parquet_path:  {eval_parquet_path}", flush=True)
    print(f"  imagenet_root: {eval_imagenet_root}", flush=True)
    print(f"  max_images:    {eval_max_images}", flush=True)
    print(f"  seed:          {eval_seed}", flush=True)
    print(f"  heatmap_size:  {eval_heatmap_size}", flush=True)
    print(f"  heatmap_sigma: {eval_heatmap_sigma}", flush=True)
    print(f"  max_seq_len:   {eval_max_seq_len}", flush=True)

    print("model settings from checkpoint:", flush=True)
    for key in ["visual_encoder", "d_model", "n_heads", "scanpath_layers",
                "fusion_layers", "dropout", "decoder_hidden_channels"]:
        print(f"  {key}: {ckpt_args.get(key, 'default')}", flush=True)

    model = build_model_from_checkpoint(checkpoint, device)

    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=eval_seed,
        max_images=eval_max_images,
        use_grouped_sampler=False,
        max_seq_len=eval_max_seq_len,
        parquet_path=eval_parquet_path,
        imagenet_root=eval_imagenet_root,
    )

    print("starting checkpoint evaluation...", flush=True)

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.eval_batches,
        heatmap_size=eval_heatmap_size,
        heatmap_sigma=eval_heatmap_sigma,
    )

    print("\nsummary", flush=True)
    print(f"steps: {int(metrics['steps'])}", flush=True)
    print(f"kl:    {metrics['kl']:.4f}", flush=True)
    print(f"ll:    {metrics['ll']:.4f}", flush=True)
    print(f"nss:   {metrics['nss']:.4f}", flush=True)
    print(f"auc:   {metrics['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
