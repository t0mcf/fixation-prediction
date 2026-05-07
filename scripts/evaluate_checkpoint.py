"""evaluate a trained scanpath model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

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

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet-path", default=default_parquet_path)
    parser.add_argument("--imagenet-root", default=default_imagenet_root)

    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=50)

    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)

    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict,
    heatmap_size: int,
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
        max_prefix_len=15,
        heatmap_size=ckpt_args.get("heatmap_size", heatmap_size),
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

        if batch_idx % 10 == 0:
            print(
                f"batch {batch_idx:04d} | "
                f"kl={value_kl.item():.4f} | "
                f"nll={value_nll.item():.4f} | "
                f"nss={value_nss.item():.4f}",
                flush=True,
            )

    return {
        "samples": total_samples,
        "kl": total_kl / total_samples,
        "nll": total_nll / total_samples,
        "nss": total_nss / total_samples,
    }


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    print("checkpoint step:", checkpoint.get("step", "unknown"), flush=True)

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        heatmap_size=args.heatmap_size,
        device=device,
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
    print(f"nll:     {metrics['nll']:.4f}", flush=True)
    print(f"nss:     {metrics['nss']:.4f}", flush=True)


if __name__ == "__main__":
    main()