#!/usr/bin/env python
"""Shared in-domain evaluator for the synthetic (ImageNet/AV) splits — ours
and DeepGaze III through ONE scoring path, written to the frozen-runner
metrics.json schema (results.<res>.{ll_uniform_img, nss_img, auc_img, ...,
per_image}).

Why this exists (2026-09-11, STATUS §4): the two legacy scripts
(evaluate_checkpoint.py, evaluate_dg3_compat.py) silently inherited
max_images=1000 from the checkpoint args and scored a class-sorted
1,000-image PREFIX of the validation split; DG3 was scored on all 5,000.
This script takes the image list explicitly, refuses to run without it,
asserts exact coverage, and scores both models with the same metric code
on the same grids.

Conventions (fixed, not configurable):
  * images       : explicit protocol list (class/file.JPEG per line); every
                   listed image must be present in the parquet, and the
                   evaluated set must equal the list exactly.
  * scanpaths    : --scanpath-ids (default 0..15 = all 16 AV paths/image).
  * geometry     : Resize(256)+CenterCrop(224) for ours (OURS_CROP_TRANSFORM=1
                   is set here BEFORE src.data.dataset is imported and then
                   asserted); DG3 uses scanpather's standard_resize = same.
  * targets      : fixations 1..L-1 of each path (the centre start fixation
                   is conditioning only); teacher forcing with the recorded
                   history; DG3 sees its usual last-4 history. For BOTH
                   models the scored target cell is derived from the ORIGINAL
                   normalized [-1,1] coordinate in the parquet with the same
                   rounding (coords_to_cells). DG3's pysaliency pipeline
                   floors coordinates to integer 224-pixels for its mask and
                   history input; the history input is left exactly as
                   trained, but the score target is NOT reconstructed from
                   that mask (floor vs round differ by one pixel for ~half
                   the fixations — review 2026-09-11). A digest of all
                   (image, path id, step, row, col) target cells per grid is
                   written to provenance so cross-model target identity can
                   be asserted from the two metrics.json files.
  * grids        : --resolutions (default 64 224). Ours: native 64, lifted
                   to 224 by mass-preserving bilinear interpolation
                   (src.eval.fair_metrics.resize_logdensity — the MIT
                   harness's function). DG3: native 224 (downsample 1);
                   pooled DOWN to 64 by area pooling (the same function
                   scanpather's MIT fair evaluator uses); at 224 it is
                   passed through untouched (no down/up round trip).
  * metrics      : LL in bits over uniform on that grid, NSS on the
                   probability map (sample std over pixels), AUC with all
                   other cells as negatives and half credit for ties —
                   numerically identical to src.training.metrics /
                   src.training.losses.ll_score (asserted on the first
                   batch), just returned per fixation instead of averaged.
  * aggregation  : fixation-weighted AND image-averaged (per-image mean,
                   then mean over images), both written; per-image rows
                   exported at full precision.
  * center bias  : none (synthetic data has a flat prior); the ig_* fields
                   are therefore equal to the ll_* fields and
                   centerbias_ll_uniform_img is 0.0 — kept so the schema
                   matches the MIT frozen runner.
  * precision    : fp32 by default. --amp reproduces the training-time
                   validation (autocast) for the LL@64 reproduction gate.

Usage
  python scripts/eval_synthetic_fair.py --model ours \
      --checkpoint runs/v2_1k_5p_ll_seed42/checkpoints/best_val_ll.pt \
      --image-list .../validation_5k.txt --name ours_1k_seed42 \
      --out-json runs/.../metrics.json
  python scripts/eval_synthetic_fair.py --model dg3 \
      --checkpoint .../dg3_1k_5sp/scanpath/best.pth \
      --config-json .../dg3_1k_5sp/scanpath/config.json \
      --image-list .../validation_5k.txt --name dg3_1k --out-json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

# geometry must be decided before src.data.dataset is imported (module-level
# transform), so this happens before any project import.
os.environ["OURS_CROP_TRANSFORM"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.fair_metrics import resize_logdensity  # noqa: E402
from src.training.losses import ll_score  # noqa: E402
from src.training.metrics import auc_score, nss_score  # noqa: E402

SCANPATHER = Path("/user/tomcosmo.fischer/u27846/repos/scanpather")
DEFAULT_PARQUET = ("/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/"
                   "2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet")
DEFAULT_IMAGENET = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"
SCHEMA_VERSION = "synthetic_fair_v1"


# ---------------------------------------------------------------------------
# per-fixation metrics (vectorised twins of the shared implementations)
# ---------------------------------------------------------------------------

def cells(xy: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = xy[:, 0], xy[:, 1]
    col = ((x + 1.0) / 2.0 * (grid - 1)).round().long().clamp(0, grid - 1)
    row = ((y + 1.0) / 2.0 * (grid - 1)).round().long().clamp(0, grid - 1)
    return row, col


def per_fixation_metrics(log_density: torch.Tensor, xy: torch.Tensor):
    """(N, G, G) log-density, (N, 2) targets in [-1, 1] -> (ll_bits, nss, auc), each (N,)."""
    N, H, W = log_density.shape
    row, col = cells(xy, H)
    idx = torch.arange(N, device=log_density.device)
    logp = log_density[idx, row, col]
    ll = (logp + math.log(H * W)) / math.log(2.0)

    prob = log_density.exp()
    flat = prob.reshape(N, -1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1).clamp_min(1e-8)          # unbiased, as torch.std default
    nss = (prob[idx, row, col] - mean) / std

    lflat = log_density.reshape(N, -1)
    target = lflat[idx, row * W + col].unsqueeze(1)
    lower = (lflat < target).sum(dim=1).float()
    equal = (lflat == target).sum(dim=1).float() - 1.0
    auc = (lower + 0.5 * equal) / (H * W - 1)
    return ll, nss, auc


def assert_metrics_match(log_density: torch.Tensor, xy: torch.Tensor) -> None:
    """The per-fixation versions must average to exactly what the shared
    implementations return (they are what every other table used)."""
    ll, nss, auc = per_fixation_metrics(log_density, xy)
    ref = (ll_score(log_density, xy).item(), nss_score(log_density, xy).item(),
           auc_score(log_density, xy).item())
    got = (ll.mean().item(), nss.mean().item(), auc.mean().item())
    for name, a, b in zip(("ll", "nss", "auc"), got, ref):
        if not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError(f"metric self-check failed for {name}: {a} vs shared {b}")


def pool_down(log_density: torch.Tensor, grid: int) -> torch.Tensor:
    """Native (N, H, W) -> (N, grid, grid) by area pooling of probability mass
    (scanpather.eval_mit1003_fair_dg3.pool_logdensity_to_grid, verbatim)."""
    prob = log_density.exp().unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(prob, (grid, grid)).squeeze(1)
    pooled = pooled.clamp_min(0)
    pooled = pooled / pooled.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(pooled.clamp_min(1e-12))


def to_grid(log_density: torch.Tensor, grid: int) -> torch.Tensor:
    native = log_density.shape[-1]
    if grid == native:
        return log_density
    if grid < native:
        return pool_down(log_density, grid)
    return resize_logdensity(log_density, grid)


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------

class Accumulator:
    def __init__(self, resolutions):
        self.res = list(resolutions)
        self.per_image = {r: OrderedDict() for r in self.res}   # image_key -> [sll, snss, sauc, n]
        self.n_fix = 0
        self.targets = {r: [] for r in self.res}   # (image_key, path_id, step, row, col)

    def add_targets(self, res: int, image_keys, path_ids, steps, xy: torch.Tensor):
        row, col = cells(xy, res)
        row, col = row.cpu().numpy(), col.cpu().numpy()
        self.targets[res].extend(zip(image_keys, path_ids, steps, row.tolist(), col.tolist()))

    def targets_digest(self, res: int) -> str:
        h = hashlib.sha256()
        for t in sorted(self.targets[res]):
            h.update(f"{t[0]}|{t[1]}|{t[2]}|{t[3]}|{t[4]}\n".encode())
        return h.hexdigest()

    def add(self, res: int, image_keys: list[str], ll, nss, auc):
        store = self.per_image[res]
        ll, nss, auc = ll.cpu().numpy(), nss.cpu().numpy(), auc.cpu().numpy()
        for k, a, b, c in zip(image_keys, ll, nss, auc):
            e = store.setdefault(k, [0.0, 0.0, 0.0, 0])
            e[0] += float(a); e[1] += float(b); e[2] += float(c); e[3] += 1

    def result(self, res: int, image_names: dict[str, str]):
        store = self.per_image[res]
        n_fix = sum(e[3] for e in store.values())
        tot = [sum(e[i] for e in store.values()) for i in range(3)]
        per_img = []
        for i, (k, e) in enumerate(store.items()):
            per_img.append({"image_id": i, "n_fixations": e[3],
                            "ll": e[0] / e[3], "ig": e[0] / e[3],
                            "nss": e[1] / e[3], "auc": e[2] / e[3],
                            "image": image_names.get(k, k)})
        img_mean = lambda key: float(np.mean([p[key] for p in per_img]))
        return {
            "ll_uniform_fix": tot[0] / n_fix,
            "ll_uniform_img": img_mean("ll"),
            "ig_centerbias_fix": tot[0] / n_fix,
            "ig_centerbias_img": img_mean("ll"),
            "centerbias_ll_uniform_img": 0.0,
            "n_fixations": int(n_fix),
            "n_images": len(per_img),
            "per_image": per_img,
            "nss_fix": tot[1] / n_fix,
            "auc_fix": tot[2] / n_fix,
            "nss_img": img_mean("nss"),
            "auc_img": img_mean("auc"),
        }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_list(path: Path) -> list[str]:
    keys = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        a = line.split("/")
        keys.append(a[-2] + "/" + a[-1])
    if len(set(keys)) != len(keys):
        raise SystemExit(f"duplicate entries in {path}")
    return keys


def _key(p: str) -> str:
    a = p.split("/")
    return a[-2] + "/" + a[-1]


# ---------------------------------------------------------------------------
# ours
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ours(args, list_keys: list[str], device: str, acc: Accumulator):
    import src.data.dataset as ds_mod
    from src.data.dataloader import image_subset_from_list, make_dataloader
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_checkpoint import build_model_from_checkpoint

    # geometry guard: the module-level transform must be the crop one
    names = [type(t).__name__ for t in ds_mod._img_transform.transforms]
    if "CenterCrop" not in names:
        raise SystemExit(f"dataset transform is {names}; OURS_CROP_TRANSFORM did not take effect")

    image_subset = image_subset_from_list(str(args.image_list), args.parquet_path)
    if len(image_subset) != len(list_keys):
        raise SystemExit(f"image list resolves to {len(image_subset)} parquet images, "
                         f"list has {len(list_keys)}")

    loader = make_dataloader(
        split="val", batch_size=args.batch_size, num_workers=args.num_workers,
        seed=0, use_grouped_sampler=False, max_samples=None, max_images=None,
        max_paths_per_image=None, max_seq_len=args.max_seq_len,
        parquet_path=args.parquet_path, imagenet_root=args.imagenet_root,
        image_subset=image_subset, epoch_subset=list(args.scanpath_ids),
    )
    df = loader.dataset.df
    # ---- coverage assertions -------------------------------------------
    got_keys = set(df["image_path"].map(_key))
    if got_keys != set(list_keys):
        raise SystemExit(f"coverage mismatch: {len(got_keys)} evaluated vs {len(list_keys)} listed; "
                         f"missing={len(set(list_keys) - got_keys)} extra={len(got_keys - set(list_keys))}")
    ids_present = sorted(df["epoch"].unique().tolist())
    if ids_present != sorted(args.scanpath_ids):
        raise SystemExit(f"scanpath ids present {ids_present} != requested {sorted(args.scanpath_ids)}")
    want_ids = sorted(args.scanpath_ids)
    bad = [k for k, g in df.groupby("image_path")["epoch"] if sorted(g.tolist()) != want_ids]
    if bad:
        raise SystemExit(f"{len(bad)} images without exactly one row per scanpath id, e.g. {bad[:3]}")
    lengths = df["locations"].map(lambda a: len(a) // 2)
    if int(lengths.min()) != args.max_seq_len or int(lengths.max()) != args.max_seq_len:
        raise SystemExit(f"scanpath lengths {int(lengths.min())}..{int(lengths.max())}, expected all {args.max_seq_len}")
    len_hist = {int(k): int(v) for k, v in lengths.value_counts().sort_index().items()}
    n_expected_targets = int(np.minimum(lengths, args.max_seq_len).sum() - len(df))
    print(f"coverage OK: {len(got_keys)} images x {len(args.scanpath_ids)} paths = {len(df)} scanpaths; "
          f"expected targets {n_expected_targets}; length histogram {len_hist}", flush=True)
    image_paths = df["image_path"].tolist()
    path_ids = df["epoch"].astype(int).tolist()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt, device)
    model.eval()
    ckpt_args = ckpt.get("args", {})
    native = int(ckpt_args.get("heatmap_size", 64))

    device_type = device.split(":")[0]
    offset, checked, t0 = 0, False, time.time()
    for bi, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        fixations = batch["fixations"].to(device)
        fl = batch["fixations_len"].to(device)
        B, S, _ = fixations.shape
        T = S - 1
        with torch.autocast(device_type=device_type, enabled=(args.amp and device_type == "cuda")):
            log_preds = model(image, fixations, fl)                 # (B, T, 64, 64)
        log_preds = log_preds.float()
        valid = (torch.arange(T, device=device).unsqueeze(0) < (fl - 1).unsqueeze(1))  # (B, T)
        vflat = valid.reshape(-1)
        lp = log_preds.reshape(B * T, native, native)[vflat]
        xy = fixations[:, 1:, :].reshape(B * T, 2)[vflat]
        sample_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T).reshape(-1)[vflat]
        step_idx = (torch.arange(T, device=device).unsqueeze(0).expand(B, T).reshape(-1)[vflat] + 1).cpu().tolist()
        si = sample_idx.cpu().tolist()
        keys = [_key(image_paths[offset + i]) for i in si]
        pids = [path_ids[offset + i] for i in si]
        offset += B
        if not checked:
            assert_metrics_match(lp, xy)
            checked = True
        if not torch.isfinite(lp).all():
            raise SystemExit("non-finite log-density in batch %d" % bi)
        for res in acc.res:
            lr = to_grid(lp, res)
            ll, nss, auc = per_fixation_metrics(lr, xy)
            acc.add(res, keys, ll, nss, auc)
            acc.add_targets(res, keys, pids, step_idx, xy)
        acc.n_fix += int(vflat.sum())
        if bi % 20 == 0:
            print(f"batch {bi:05d} | fixations so far {acc.n_fix} | {time.time() - t0:.0f}s", flush=True)
    if acc.n_fix != n_expected_targets:
        raise SystemExit(f"target count {acc.n_fix} != expected {n_expected_targets}")
    names = {_key(p): Path(p).name for p in image_paths}
    return names, {"n_scanpaths": int(len(df)), "scanpath_ids": sorted(args.scanpath_ids),
                   "length_histogram": len_hist, "native_grid": native,
                   "checkpoint_args_max_images_ignored": ckpt_args.get("max_images")}


# ---------------------------------------------------------------------------
# DeepGaze III
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_dg3(args, list_keys: list[str], device: str, acc: Accumulator):
    sys.path.insert(0, str(SCANPATHER))
    sys.path.insert(0, str(SCANPATHER / "models" / "DeepGaze"))
    from deepgaze_pytorch.data import FixationDataset, ImageDatasetSampler
    from scanpath.data.parquet_to_pysaliency import parquet_to_pysaliency
    from scanpath.deepgaze_engine import preprocess_to_resized_filestimuli, sample_transform
    from scanpath.eval_imagenet_fullval import load_model, _mask_to_xy
    from scanpath.train_deepgaze import _ZeroCenterbias

    if sorted(args.scanpath_ids) != list(range(16)):
        raise SystemExit("DG3 path: scanpath ids must be 0..15 (parquet_to_pysaliency has no id filter)")

    # Original normalized coordinates, keyed by (image key, path id): the
    # score targets for DG3 come from here, exactly as for ours — never from
    # the floored 224-pixel mask the pysaliency pipeline builds.
    import pandas as pd
    df = pd.read_parquet(args.parquet_path, engine="pyarrow",
                         columns=["image_path", "epoch", "locations"])
    df = df[df["image_path"].map(_key).isin(set(list_keys))].reset_index(drop=True)
    orig = {}
    for k, e, loc in zip(df["image_path"].map(_key), df["epoch"].astype(int), df["locations"]):
        orig[(k, e)] = np.asarray(loc, dtype=np.float32).reshape(-1, 2)
    del df
    want_ids = list(range(16))
    per_img = {}
    for (k, e) in orig:
        per_img.setdefault(k, []).append(e)
    bad = [k for k, ids in per_img.items() if sorted(ids) != want_ids]
    if set(per_img) != set(list_keys) or bad:
        raise SystemExit(f"parquet coverage: {len(per_img)} images, {len(bad)} with a wrong id multiset")
    if any(len(v) != args.max_seq_len for v in orig.values()):
        raise SystemExit("not every AV scanpath has exactly max_seq_len fixations")

    _, _, stimuli, fixations = parquet_to_pysaliency(
        parquet_path=Path(args.parquet_path), seed=3141,
        image_base_dir=Path(args.imagenet_root) / "train_images", target_size=224,
        train_n_images=1, val_n_images=len(list_keys), train_n_scanpaths=1,
        val_n_scanpaths=None, val_image_list=Path(args.image_list),
    )
    stimuli = preprocess_to_resized_filestimuli(stimuli)
    filenames = [str(f) for f in stimuli.filenames]
    got_keys = {_key(f) for f in filenames}
    if got_keys != set(list_keys):
        raise SystemExit(f"coverage mismatch: {len(got_keys)} stimuli vs {len(list_keys)} listed")
    n_targets = len(fixations)
    n_paths = len(set(zip(fixations.n.tolist(), fixations.subjects.tolist()))) if hasattr(fixations, "subjects") else None
    print(f"coverage OK: {len(filenames)} images; {n_targets} fixation targets"
          + (f"; {n_paths} scanpaths" if n_paths else ""), flush=True)

    class KeyedFixationDataset(FixationDataset):
        def __getitem__(self, key):
            d = super().__getitem__(key)
            d["sample_key"] = int(key)
            return d

    # exactly the kwargs scanpather.deepgaze_engine.make_scanpath_loader uses
    # (4-step history; missing early history filled with NaN as in training)
    dataset = KeyedFixationDataset(
        stimuli=stimuli, fixations=fixations, centerbias_model=_ZeroCenterbias(),
        transform=sample_transform, average="image", lmdb_path=None,
        included_fixations=[-1, -2, -3, -4], allow_missing_fixations=True,
        cache_image_data=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=ImageDatasetSampler(dataset, batch_size=args.batch_size, shuffle=False),
        pin_memory=True, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    model = load_model(Path(args.checkpoint), device,
                       Path(args.config_json) if args.config_json else None)
    model.eval()
    fix_n, fix_subj, fix_len = fixations.n, fixations.subjects, fixations.lengths
    fix_xint, fix_yint = fixations.x_int, fixations.y_int
    if n_targets != len(list_keys) * 16 * (args.max_seq_len - 1):
        raise SystemExit(f"DG3 target count {n_targets} != {len(list_keys)}*16*{args.max_seq_len - 1}")

    device_type = device.split(":")[0]
    native_seen, t0 = None, time.time()
    legacy_mismatch = [0, 0]
    for bi, batch in enumerate(loader):
        keys_idx = batch.pop("sample_key").tolist()
        image = batch.pop("image").to(device)
        centerbias = batch.pop("centerbias").to(device)
        fixation_mask = batch.pop("fixation_mask")
        if fixation_mask.is_sparse:
            fixation_mask = fixation_mask.to_dense()
        fixation_mask = fixation_mask.to(device)
        x_hist = batch.pop("x_hist", torch.tensor([])).to(device)
        y_hist = batch.pop("y_hist", torch.tensor([])).to(device)
        batch.pop("weight", None)
        durations = batch.pop("durations", torch.tensor([])).to(device)
        kwargs = {k: v.to(device) for k, v in dict(batch).items()}
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(args.amp and device_type == "cuda")):
            log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist,
                                durations=durations, **kwargs)
        log_density = log_density.float()
        if native_seen is None:
            native_seen = tuple(log_density.shape[-2:])
            print(f"DG3 native output grid: {native_seen}", flush=True)
            if native_seen != (224, 224):
                raise SystemExit(f"DG3 native output {native_seen}, expected (224, 224)")
        if not torch.isfinite(log_density).all():
            raise SystemExit("non-finite DG3 log-density in batch %d" % bi)
        # identity: sample -> (image, path id, step) -> original coordinate
        keys, pids, steps, xy_np = [], [], [], np.empty((len(keys_idx), 2), np.float32)
        for j, k in enumerate(keys_idx):
            kk = _key(filenames[int(fix_n[k])]); e = int(fix_subj[k]); st = int(fix_len[k])
            loc = orig[(kk, e)][st]
            # consistency with the pysaliency pipeline: its floored pixel
            # must be the floor of the same original coordinate
            px = int(min(max((loc[0] + 1.0) * 0.5 * 224, 0.0), 223.0))
            py = int(min(max((loc[1] + 1.0) * 0.5 * 224, 0.0), 223.0))
            if px != int(fix_xint[k]) or py != int(fix_yint[k]):
                raise SystemExit(f"target identity mismatch at sample {k}: parquet ({px},{py}) vs mask ({int(fix_xint[k])},{int(fix_yint[k])})")
            keys.append(kk); pids.append(e); steps.append(st); xy_np[j] = loc
        xy = torch.from_numpy(xy_np).to(device)
        xy_mask = _mask_to_xy(fixation_mask).to(device)      # legacy diagnostic only
        for res in acc.res:
            lr = to_grid(log_density, res)
            ll, nss, auc = per_fixation_metrics(lr, xy)
            acc.add(res, keys, ll, nss, auc)
            acc.add_targets(res, keys, pids, steps, xy)
        # how often the legacy mask-derived cell differs from the original one
        r1, c1 = cells(xy, 224); r2, c2 = cells(xy_mask, 224)
        legacy_mismatch[0] += int(((r1 != r2) | (c1 != c2)).sum()); legacy_mismatch[1] += len(keys)
        acc.n_fix += log_density.shape[0]
        if bi % 20 == 0:
            print(f"batch {bi:05d} | fixations so far {acc.n_fix} | {time.time() - t0:.0f}s", flush=True)
    if acc.n_fix != n_targets:
        raise SystemExit(f"target count {acc.n_fix} != expected {n_targets}")
    names = {_key(f): Path(f).name for f in filenames}
    print(f"legacy mask-derived target cell differs from original-coordinate cell at 224 "
          f"for {legacy_mismatch[0]} / {legacy_mismatch[1]} fixations (diagnostic only)", flush=True)
    return names, {"n_scanpaths": n_paths, "scanpath_ids": list(range(16)),
                   "native_grid": native_seen[0],
                   "legacy_mask_target_mismatch_224": legacy_mismatch}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ours", "dg3"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-json", default=None, help="DG3: stage config.json (readout factors)")
    ap.add_argument("--image-list", required=True, type=Path,
                    help="protocol split file; REQUIRED — there is no default split")
    ap.add_argument("--scanpath-ids", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[64, 224])
    ap.add_argument("--max-seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="autocast (reproduces training-time validation); default fp32")
    ap.add_argument("--parquet-path", default=DEFAULT_PARQUET)
    ap.add_argument("--imagenet-root", default=DEFAULT_IMAGENET)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    if "test" in args.image_list.name and os.environ.get("SYNTHETIC_TEST_OK") != "1":
        raise SystemExit("refusing a test-split list without SYNTHETIC_TEST_OK=1 (test day only)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)
    list_keys = read_list(args.image_list)
    acc = Accumulator(args.resolutions)
    t0 = time.time()
    if args.model == "ours":
        names, coverage = run_ours(args, list_keys, device, acc)
    else:
        names, coverage = run_dg3(args, list_keys, device, acc)

    results = {str(r): acc.result(r, names) for r in args.resolutions}
    exp_per_img = len(args.scanpath_ids) * (args.max_seq_len - 1)
    for r in args.resolutions:
        counts = {p["n_fixations"] for p in results[str(r)]["per_image"]}
        if counts != {exp_per_img} or results[str(r)]["n_images"] != len(list_keys):
            raise SystemExit(f"per-image target counts {counts} / n_images {results[str(r)]['n_images']} "
                             f"!= {exp_per_img} / {len(list_keys)}")
        for p in results[str(r)]["per_image"]:
            if not all(math.isfinite(p[k]) for k in ("ll", "nss", "auc")):
                raise SystemExit(f"non-finite per-image metric for {p['image']}")
    target_digests = {str(r): acc.targets_digest(r) for r in args.resolutions}
    out = {
        "schema_version": SCHEMA_VERSION,
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "provenance": {
            "checkpoint_sha256": sha256(Path(args.checkpoint)),
            "image_list_sha256": sha256(args.image_list),
            "n_images_listed": len(list_keys),
            "torch_version": torch.__version__,
            "device": device,
            "amp": bool(args.amp),
            "wall_seconds": round(time.time() - t0, 1),
            "coverage": coverage,
            "target_cells_sha256": target_digests,
        },
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1) + "\n")
    print("\nsummary", flush=True)
    for r in args.resolutions:
        m = results[str(r)]
        print(f"  grid {r:4d} | LL_img {m['ll_uniform_img']:.4f} LL_fix {m['ll_uniform_fix']:.4f} | "
              f"NSS_img {m['nss_img']:.4f} | AUC_img {m['auc_img']:.4f} | "
              f"n_fix {m['n_fixations']} n_img {m['n_images']}", flush=True)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
