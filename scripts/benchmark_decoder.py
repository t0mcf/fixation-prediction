"""
Isolated timing benchmark for the three heatmap-decoder upsampling variants.

Why isolated rather than wall-clock from training runs: training time on a
shared node is contaminated by data loading and by other jobs, and the decoder
is only one part of the step. This times the decoder alone (forward + backward)
on synthetic GPU tensors, which is what the design decision actually turned on
(the bilinear kernel was ~100x off the A100 memory
bandwidth limit at these shapes).

Shapes match training: the decoder is called on (B*T, N, d_model), so with
batch_size=128 scanpaths and T=15 prediction steps the effective batch is large.

  python scripts/benchmark_decoder.py
  python scripts/benchmark_decoder.py --batch 480 --reps 50
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.dino_scanpather import SpatialHeatmapDecoder

VARIANTS = ["nearest", "bilinear", "transposed"]


def bench(variant: str, batch: int, d_model: int, n_patches: int,
          reps: int, warmup: int, device: str) -> dict:
    dec = SpatialHeatmapDecoder(d_model=d_model, upsample=variant).to(device)
    x = torch.randn(batch, n_patches, d_model, device=device, requires_grad=True)

    def step():
        out = dec(x)
        loss = out.square().mean()
        loss.backward()
        dec.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps

    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    n_params = sum(p.numel() for p in dec.parameters())
    return {"variant": variant, "ms": dt * 1e3, "peak_mib": peak, "params": n_params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=480,
                    help="effective batch = scanpaths x prediction steps (default 32x15)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-patches", type=int, default=256)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"decoder input: ({args.batch}, {args.n_patches}, {args.d_model}) "
          f"-> (B, 64, 64), fwd+bwd, {args.reps} reps after {args.warmup} warmup\n")

    rows = [bench(v, args.batch, args.d_model, args.n_patches,
                  args.reps, args.warmup, device) for v in VARIANTS]
    base = next(r["ms"] for r in rows if r["variant"] == "nearest")

    print(f"{'variant':<12}{'ms/step':>10}{'vs nearest':>12}{'peak MiB':>11}{'params':>12}")
    print("-" * 57)
    for r in rows:
        print(f"{r['variant']:<12}{r['ms']:>10.2f}{r['ms']/base:>11.2f}x"
              f"{r['peak_mib']:>11.0f}{r['params']:>12,}")
    print("\nnote: 'transposed' has fewer parameters because ConvTranspose2d(k=2) "
          "replaces the Upsample+Conv2d(k=3) pair -- it is not parameter-matched.")


if __name__ == "__main__":
    main()
