"""
Profile ScanpathModel.

Uses the real model.forward() (no manual reimplementation drift), the real
training loss (NLL or KL), and times target heatmap creation separately.
Per-module forward and backward timing via PyTorch hooks with cuda.synchronize
fences. Pass 1 is torch.profiler for the raw op table; pass 2 is hook-based
per-module timing.

Out of scope: DataLoader cost, validation, checkpointing, W&B logging. Those
need a separate script with the real make_dataloader.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.profiler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.models.dino_scanpather import ScanpathModel
from src.training.losses import (
    make_gaussian_heatmaps_batch,
    multi_step_kl_loss,
    multi_step_ll_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scanpath-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=4)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--decoder-hidden-channels", type=int, default=256)
    parser.add_argument("--decoder-upsample", default="nearest",
                        choices=["nearest", "bilinear", "transposed"],
                        help="heatmap-decoder upsampling variant; lets this script "
                             "report the decoder's share of the step for each option "
                             "in the decoder ablation.")
    parser.add_argument("--paths-per-image", type=int, default=16,
                        help="grouped-sampler optimization: run encoder on B//P unique images")
    parser.add_argument("--loss", choices=["ll", "kl"], default="ll",
                        help="real training loss (matches training)")
    parser.add_argument("--amp", action="store_true", help="use AMP (matches training)")
    parser.add_argument("--bidirectional-fusion", action="store_true")
    parser.add_argument("--patch-pos-embed", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/profile")
    parser.add_argument("--export-trace", action="store_true")
    return parser.parse_args()


def make_fake_batch(batch_size: int, seq_len: int, device: str):
    image = torch.randn(batch_size, 3, 224, 224, device=device)
    fixations = torch.rand(batch_size, seq_len, 2, device=device) * 2 - 1
    # full lengths = worst-case compute. Variable lengths only affect masking,
    # not the underlying tensor kernel sizes.
    fixations_len = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    return image, fixations, fixations_len


class ModuleTimer:
    """
    Times forward and backward of named modules via PyTorch hooks.
    cuda.synchronize() fences inside each hook so timings reflect actual GPU work.
    """

    def __init__(self, named_modules: dict, device: str):
        self.named = named_modules
        self.device = device
        self.times: dict = defaultdict(lambda: {"forward": 0.0, "backward": 0.0})
        self._fwd_start: dict = {}
        self._bwd_start: dict = {}
        self._handles: list = []

        for name, mod in named_modules.items():
            self._handles.append(mod.register_forward_pre_hook(self._make_pre_fwd(name)))
            self._handles.append(mod.register_forward_hook(self._make_post_fwd(name)))
            try:
                self._handles.append(mod.register_full_backward_pre_hook(self._make_pre_bwd(name)))
                self._handles.append(mod.register_full_backward_hook(self._make_post_bwd(name)))
            except (AttributeError, RuntimeError) as e:
                print(f"  warning: could not install backward hooks for {name}: {e}", flush=True)

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _make_pre_fwd(self, name):
        def hook(module, args):
            self._sync()
            self._fwd_start[name] = time.perf_counter()
        return hook

    def _make_post_fwd(self, name):
        def hook(module, args, output):
            self._sync()
            if name in self._fwd_start:
                self.times[name]["forward"] += time.perf_counter() - self._fwd_start[name]
                del self._fwd_start[name]
        return hook

    def _make_pre_bwd(self, name):
        def hook(module, grad_output):
            self._sync()
            self._bwd_start[name] = time.perf_counter()
        return hook

    def _make_post_bwd(self, name):
        def hook(module, grad_input, grad_output):
            self._sync()
            if name in self._bwd_start:
                self.times[name]["backward"] += time.perf_counter() - self._bwd_start[name]
                del self._bwd_start[name]
        return hook

    def reset(self):
        self.times = defaultdict(lambda: {"forward": 0.0, "backward": 0.0})
        self._fwd_start.clear()
        self._bwd_start.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"seq_len: {args.seq_len}  (T = {args.seq_len - 1} prediction steps)", flush=True)
    print(f"fusion_layers: {args.fusion_layers}", flush=True)
    print(f"paths_per_image: {args.paths_per_image}", flush=True)
    print(f"loss: {args.loss}", flush=True)
    print(f"amp: {use_amp}", flush=True)
    print(f"bidirectional_fusion: {args.bidirectional_fusion}", flush=True)
    print(f"patch_pos_embed: {args.patch_pos_embed}", flush=True)
    print(f"warmup_steps: {args.warmup_steps}", flush=True)
    print(f"profile_steps: {args.profile_steps}", flush=True)

    model = ScanpathModel(
        visual_encoder="dinov2_base",
        pretrained_encoder=True,
        img_size=224,
        d_model=args.d_model,
        n_heads=args.n_heads,
        scanpath_layers=args.scanpath_layers,
        fusion_layers=args.fusion_layers,
        max_seq_len=args.seq_len,
        heatmap_size=args.heatmap_size,
        dropout=0.0,
        decoder_hidden_channels=args.decoder_hidden_channels,
        decoder_upsample=args.decoder_upsample,
        use_bidirectional_fusion=args.bidirectional_fusion,
        use_patch_pos_embed=args.patch_pos_embed,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\ntrainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"total params:     {sum(p.numel() for p in model.parameters()):,}", flush=True)

    def compute_loss(log_preds, target_fixations, fixations_len, target_heatmaps):
        log_preds = log_preds.float()
        if args.loss == "kl":
            return multi_step_kl_loss(log_preds, target_heatmaps, fixations_len)
        return -multi_step_ll_score(log_preds, target_fixations, fixations_len)

    def run_real_step():
        image, fixations, fixations_len = make_fake_batch(args.batch_size, args.seq_len, device)
        target_fixations = fixations[:, 1:, :]
        target_heatmaps = None
        if args.loss == "kl":
            target_heatmaps = make_gaussian_heatmaps_batch(
                target_fixations, args.heatmap_size, args.heatmap_sigma,
            ).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=use_amp):
            log_preds = model(image, fixations, fixations_len,
                              paths_per_image=args.paths_per_image)
        loss = compute_loss(log_preds, target_fixations, fixations_len, target_heatmaps)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ---- warmup ----
    print(f"\nrunning {args.warmup_steps} warmup steps...", flush=True)
    model.train()
    for _ in range(args.warmup_steps):
        run_real_step()
    if device == "cuda":
        torch.cuda.synchronize()
    print("warmup done.\n", flush=True)

    # ============================================================================
    # PASS 1: torch.profiler for raw op table (NO timing hooks installed)
    # ============================================================================
    print("=" * 80, flush=True)
    print("PASS 1: torch.profiler — raw op table", flush=True)
    print("=" * 80, flush=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities, record_shapes=True, with_stack=False,
    ) as prof:
        for step in range(args.profile_steps):
            with torch.profiler.record_function(f"step_{step}"):
                run_real_step()
            if device == "cuda":
                torch.cuda.synchronize()

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25), flush=True)

    if args.export_trace:
        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"chrome trace saved to {trace_path}", flush=True)

    # ============================================================================
    # PASS 2: hook-based per-module timing + section-level wall-clock
    # ============================================================================
    print("\n" + "=" * 80, flush=True)
    print("PASS 2: per-module timing (hooks + cuda.synchronize fences)", flush=True)
    print("=" * 80, flush=True)

    modules_to_time: dict = {
        "image_encoder":    model.image_encoder,
        "image_adapter":    model.image_adapter,
        "scanpath_encoder": model.scanpath_encoder,
        "decoder":          model.decoder,
    }
    if getattr(model, "use_patch_pos_embed", False):
        modules_to_time["patch_pos_embed"] = model.patch_pos_embed
    for i, layer in enumerate(model.fusion):
        modules_to_time[f"fusion_{i}"] = layer

    timer = ModuleTimer(modules_to_time, device)
    extern_times: dict = defaultdict(float)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.profile_steps):
        image, fixations, fixations_len = make_fake_batch(args.batch_size, args.seq_len, device)
        target_fixations = fixations[:, 1:, :]

        sync(); t0 = time.perf_counter()

        # 1. target heatmap creation (KL only)
        if args.loss == "kl":
            target_heatmaps = make_gaussian_heatmaps_batch(
                target_fixations, args.heatmap_size, args.heatmap_sigma,
            ).float()
        else:
            target_heatmaps = None
        sync(); t1 = time.perf_counter()
        extern_times["target_heatmap"] += t1 - t0

        # 2. zero_grad
        optimizer.zero_grad(set_to_none=True)
        sync(); t2 = time.perf_counter()
        extern_times["zero_grad"] += t2 - t1

        # 3. forward (hooks fire during this)
        with torch.autocast(device_type=device, enabled=use_amp):
            log_preds = model(image, fixations, fixations_len,
                              paths_per_image=args.paths_per_image)
        sync(); t3 = time.perf_counter()
        extern_times["forward_total"] += t3 - t2

        # 4. loss
        loss = compute_loss(log_preds, target_fixations, fixations_len, target_heatmaps)
        sync(); t4 = time.perf_counter()
        extern_times["loss"] += t4 - t3

        # 5. backward (hooks fire during this)
        scaler.scale(loss).backward()
        sync(); t5 = time.perf_counter()
        extern_times["backward_total"] += t5 - t4

        # 6. optimizer step
        scaler.step(optimizer)
        scaler.update()
        sync(); t6 = time.perf_counter()
        extern_times["optimizer"] += t6 - t5

        extern_times["full_step"] += t6 - t0

    n = args.profile_steps

    # ---- per-module breakdown ----
    print(f"\nper-module forward / backward (averaged over {n} steps):", flush=True)
    print(f"{'module':<28} {'forward':>12} {'backward':>12} {'fwd+bwd':>12}", flush=True)
    print("-" * 68, flush=True)

    base_order = ["image_encoder", "image_adapter"]
    if "patch_pos_embed" in timer.times:
        base_order.append("patch_pos_embed")
    base_order.append("scanpath_encoder")

    for name in base_order:
        fwd = timer.times[name]["forward"] / n * 1e3
        bwd = timer.times[name]["backward"] / n * 1e3
        print(f"  {name:<26} {fwd:>9.2f} ms {bwd:>9.2f} ms {fwd + bwd:>9.2f} ms", flush=True)

    fusion_fwd_total = 0.0
    fusion_bwd_total = 0.0
    for i in range(args.fusion_layers):
        name = f"fusion_{i}"
        fwd = timer.times[name]["forward"] / n * 1e3
        bwd = timer.times[name]["backward"] / n * 1e3
        fusion_fwd_total += fwd
        fusion_bwd_total += bwd
        print(f"    {name:<24} {fwd:>9.2f} ms {bwd:>9.2f} ms {fwd + bwd:>9.2f} ms", flush=True)
    print(f"  {'fusion (sum)':<26} {fusion_fwd_total:>9.2f} ms {fusion_bwd_total:>9.2f} ms "
          f"{fusion_fwd_total + fusion_bwd_total:>9.2f} ms", flush=True)

    fwd = timer.times["decoder"]["forward"] / n * 1e3
    bwd = timer.times["decoder"]["backward"] / n * 1e3
    print(f"  {'decoder':<26} {fwd:>9.2f} ms {bwd:>9.2f} ms {fwd + bwd:>9.2f} ms", flush=True)

    # ---- section-level wall-clock breakdown ----
    print(f"\nsection-level wall-clock (averaged over {n} steps):", flush=True)
    print(f"{'section':<28} {'time':>12} {'% of step':>12}", flush=True)
    print("-" * 55, flush=True)
    full_step_ms = extern_times["full_step"] / n * 1e3
    for section in ["target_heatmap", "zero_grad", "forward_total", "loss", "backward_total", "optimizer"]:
        if extern_times[section] > 0:
            ms = extern_times[section] / n * 1e3
            pct = 100 * ms / full_step_ms if full_step_ms > 0 else 0
            print(f"  {section:<26} {ms:>9.2f} ms {pct:>9.1f}%", flush=True)
    print("-" * 55, flush=True)
    print(f"  {'full step (with sync)':<26} {full_step_ms:>9.2f} ms     100.0%", flush=True)

    print("", flush=True)
    print("Notes:", flush=True)
    print("  - Per-module hooks add cuda.synchronize() overhead, so step time here is", flush=True)
    print("    SLOWER than real training. Relative breakdown is accurate, absolute is not.", flush=True)
    print("  - Sum of per-module forwards < forward_total because of autocast/Python overhead", flush=True)
    print("    that isn't bounded by any single module.", flush=True)
    print("  - image_encoder backward = 0 because @torch.no_grad() blocks gradient flow.", flush=True)
    print("  - DataLoader cost is NOT included (fake tensors generated on GPU).", flush=True)

    timer.remove()


if __name__ == "__main__":
    main()
