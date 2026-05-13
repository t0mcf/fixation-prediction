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
from src.training.losses import kl_loss, ll_score
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
    parser.add_argument("--eval-batches", type=int, default=50)

    parser.add_argument("--heatmap-size", type=int, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=None)

    return parser.parse_args()


def get_ckpt_arg(
    ckpt_args: dict[str, Any],
    cli_value: Any,
    name: str,
    fallback: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(name, fallback)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: str,
) -> ScanpathModel:
    ckpt_args = checkpoint.get("args", {})

    model = ScanpathModel(
        visual_encoder=ckpt_args.get("visual_encoder", "dinov2_base"),
        pretrained_encoder=not ckpt_args.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=ckpt_args.get("d_model", 256),
        n_heads=ckpt_args.get("n_heads", 8),
        scanpath_layers=ckpt_args.get("scanpath_layers", 2),
        fusion_layers=ckpt_args.get("fusion_layers", 1),
        max_prefix_len=ckpt_args.get("max_prefix_len", 15),
        heatmap_size=ckpt_args.get("heatmap_size", 64),
        dropout=ckpt_args.get("dropout", 0.1),
        decoder_hidden_channels=ckpt_args.get("decoder_hidden_channels", 256),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()

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

        log_pred = model(image, prefix, prefix_len)

        log_pred = log_pred.float()
        heatmap = heatmap.float()

        batch_size = image.shape[0]

        value_kl = kl_loss(log_pred, heatmap)
        value_ll = ll_score(log_pred, target_xy)
        value_nss = nss_score(log_pred, target_xy)
        value_auc = auc_score(log_pred, target_xy)

        total_kl += value_kl.item() * batch_size
        total_ll += value_ll.item() * batch_size
        total_nss += value_nss.item() * batch_size
        total_auc += value_auc.item() * batch_size
        total_samples += batch_size

        if batch_idx % 10 == 0:
            print(
                f"batch {batch_idx:04d} | "
                f"kl={value_kl.item():.4f} | "
                f"ll={value_ll.item():.4f} | "
                f"nss={value_nss.item():.4f} | "
                f"auc={value_auc.item():.4f}",
                flush=True,
            )

    if total_samples == 0:
        return {
            "samples": 0,
            "kl": float("nan"),
            "ll": float("nan"),
            "nss": float("nan"),
            "auc": float("nan"),
        }

    return {
        "samples": total_samples,
        "kl": total_kl / total_samples,
        "ll": total_ll / total_samples,
        "nss": total_nss / total_samples,
        "auc": total_auc / total_samples,
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

    eval_parquet_path = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.parquet_path,
        name="parquet_path",
        fallback=default_parquet_path,
    )
    eval_imagenet_root = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.imagenet_root,
        name="imagenet_root",
        fallback=default_imagenet_root,
    )
    eval_max_images = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.max_images,
        name="max_images",
        fallback=1000,
    )
    eval_seed = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.seed,
        name="seed",
        fallback=42,
    )
    eval_heatmap_size = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.heatmap_size,
        name="heatmap_size",
        fallback=64,
    )
    eval_heatmap_sigma = get_ckpt_arg(
        ckpt_args=ckpt_args,
        cli_value=args.heatmap_sigma,
        name="heatmap_sigma",
        fallback=2.0,
    )
    eval_max_prefix_len = ckpt_args.get("max_prefix_len", 15)

    print("resolved eval settings:", flush=True)
    print(f"  parquet_path: {eval_parquet_path}", flush=True)
    print(f"  imagenet_root: {eval_imagenet_root}", flush=True)
    print(f"  max_images: {eval_max_images}", flush=True)
    print(f"  seed: {eval_seed}", flush=True)
    print(f"  heatmap_size: {eval_heatmap_size}", flush=True)
    print(f"  heatmap_sigma: {eval_heatmap_sigma}", flush=True)
    print(f"  max_prefix_len: {eval_max_prefix_len}", flush=True)

    print("model settings from checkpoint:", flush=True)
    print(f"  visual_encoder: {ckpt_args.get('visual_encoder', 'dinov2_base')}", flush=True)
    print(f"  no_pretrained_encoder: {ckpt_args.get('no_pretrained_encoder', False)}", flush=True)
    print(f"  d_model: {ckpt_args.get('d_model', 256)}", flush=True)
    print(f"  n_heads: {ckpt_args.get('n_heads', 8)}", flush=True)
    print(f"  scanpath_layers: {ckpt_args.get('scanpath_layers', 2)}", flush=True)
    print(f"  fusion_layers: {ckpt_args.get('fusion_layers', 1)}", flush=True)
    print(f"  dropout: {ckpt_args.get('dropout', 0.1)}", flush=True)
    print(f"  decoder_hidden_channels: {ckpt_args.get('decoder_hidden_channels', 256)}", flush=True)

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    val_loader = make_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=eval_seed,
        max_images=eval_max_images,
        use_grouped_sampler=False,
        heatmap_sigma=eval_heatmap_sigma,
        heatmap_size=eval_heatmap_size,
        max_prefix_len=eval_max_prefix_len,
        parquet_path=eval_parquet_path,
        imagenet_root=eval_imagenet_root,
    )

    print("starting checkpoint evaluation...", flush=True)

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.eval_batches,
    )

    print("\nsummary", flush=True)
    print(f"samples: {int(metrics['samples'])}", flush=True)
    print(f"kl:      {metrics['kl']:.4f}", flush=True)
    print(f"ll:      {metrics['ll']:.4f}", flush=True)
    print(f"nss:     {metrics['nss']:.4f}", flush=True)
    print(f"auc:     {metrics['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()