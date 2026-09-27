"""Benchmark: precomputed vs on-the-fly DINOv2 feature loading (with training step).

Measures wall-clock time per batch including model forward + backward pass,
comparing precomputed features vs on-the-fly extraction. Run on a GPU node.

Usage:
    python tools/benchmark_features.py --data-dir ./data/oursynth_smoke --n-batches 50
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.components.mit1003.mit1003 import MIT1003Dataset
from src.data.components.onthefly_features import OnTheFlyFeatureDataset
from src.model.components.model import DiT


def collate_fn(batch):
    batch = [b for b in batch if b['scanpath'] is not None]
    if len(batch) == 0:
        return None
    return {
        'img': torch.stack([b['img'] for b in batch]),
        'scanpath': torch.nn.utils.rnn.pad_sequence(
            [b['scanpath'] for b in batch], batch_first=True
        ),
    }


def make_model(device):
    model = DiT(
        max_len=16,
        spatial_dim=(37, 37),
        in_channels=4,
        scanpath_emb_size=256,
        hidden_size=512,
        img_feature_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ).to(device)
    model.train()
    return model


def benchmark(loader, n_batches, label, device, model, optimizer):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device)
        scanpath = batch['scanpath'].to(device)
        B = img.shape[0]

        # simulate diffusion training step: random timesteps, noised scanpath input
        t = torch.randint(0, 1000, (B,), device=device)
        # scanpath is (B, seq_len, 2), model expects (B, max_len, 3) for x
        seq_len = scanpath.shape[1]
        x = torch.randn(B, 16, 3, device=device)

        out = model(x, t, img)
        loss = out.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1
        if count >= n_batches:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_batch = elapsed / count
    print(f"[{label}] {count} batches in {elapsed:.2f}s  ({per_batch*1000:.1f} ms/batch)")
    return elapsed, per_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/oursynth_smoke")
    parser.add_argument("--images-dir", default=None,
                        help="path to images (default: {data-dir}/images)")
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda"

    data_dir = Path(args.data_dir)
    json_candidates = list(data_dir.glob("*_fixations_train.json"))
    if not json_candidates:
        raise FileNotFoundError(f"No *_fixations_train.json in {data_dir}")

    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Model params: {n_params/1e6:.1f}M")
    print(f"Benchmarking {args.n_batches} batches, batch_size={args.batch_size}\n")

    # --- precomputed ---
    ds_pre = MIT1003Dataset(
        name="bench_pre", root_path=args.data_dir, task="", split="train",
        num_subjects=5, img_features_dir="dinov2_base_timm_image_features",
    )
    loader_pre = DataLoader(
        ds_pre, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # --- on-the-fly ---
    ds_base = MIT1003Dataset(
        name="bench_otf", root_path=args.data_dir, task="", split="train",
        num_subjects=5, img_features_dir="dinov2_base_timm_image_features",
    )
    images_dir = args.images_dir or str(data_dir / "images")
    ds_otf = OnTheFlyFeatureDataset(
        ds_base, images_dir=images_dir, device=device,
    )
    loader_otf = DataLoader(
        ds_otf, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # warmup
    print("Warming up...")
    for batch in loader_otf:
        if batch is not None:
            img = batch['img'].to(device)
            x = torch.randn(img.shape[0], 16, 3, device=device)
            t = torch.randint(0, 1000, (img.shape[0],), device=device)
            out = model(x, t, img)
            out.mean().backward()
            optimizer.zero_grad()
            break
    print()

    # benchmark
    t_pre, per_pre = benchmark(loader_pre, args.n_batches, "precomputed", device, model, optimizer)
    t_otf, per_otf = benchmark(loader_otf, args.n_batches, "on-the-fly ", device, model, optimizer)

    print(f"\nSlowdown factor (full training step): {per_otf/per_pre:.2f}x")
    print(f"  precomputed: {per_pre*1000:.1f} ms/batch (data + train)")
    print(f"  on-the-fly:  {per_otf*1000:.1f} ms/batch (data + train)")


if __name__ == "__main__":
    main()
