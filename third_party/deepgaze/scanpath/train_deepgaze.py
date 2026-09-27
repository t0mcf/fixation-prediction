"""
scanpath/train_deepgaze.py
==========================
Single entry point for DeepGaze III training.

Mode is driven by `training.mode` in the YAML config:

  imagenet  — two-stage training on synthetic scanpaths from a parquet file
  salicon   — one-stage spatial pretraining on SALICON
  mit1003   — three-stage, per-fold training on MIT1003 (needs SALICON checkpoint)

Usage:
  python -m scanpath.train_deepgaze --config scanpath/configs/imagenet.yaml
  python -m scanpath.train_deepgaze --config scanpath/configs/salicon.yaml
  python -m scanpath.train_deepgaze --config scanpath/configs/mit1003.yaml

CLI args override matching leaf keys in the config, e.g.:
  python -m scanpath.train_deepgaze --config scanpath/configs/mit1003.yaml --crossval_folds 1
"""
import sys
import os
import time
import uuid
import json
import gc

# Make models/DeepGaze importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "DeepGaze"))

from pathlib import Path

import numpy as np
import torch
import warnings

# NumPy 2.0 removed np.in1d; pysaliency still uses it internally.
if not hasattr(np, 'in1d'):
    np.in1d = lambda ar1, ar2, **kw: np.isin(ar1, ar2, **kw).ravel()

# np.in1d was removed in NumPy 2.0; patch it back so pysaliency keeps working.
if not hasattr(np, "in1d"):
    warnings.warn(
        "np.in1d not found (NumPy >= 2.0) — aliasing to np.isin for pysaliency compatibility.",
        DeprecationWarning,
        stacklevel=1,
    )
    np.in1d = np.isin  # type: ignore[attr-defined]

import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import wandb

from deepgaze_pytorch.training import _train
import deepgaze_pytorch.training as _dg_training
from deepgaze_pytorch.modules import DeepGazeII as _DeepGazeII
from deepgaze_pytorch.metrics import log_likelihood as _log_likelihood, nss as _nss, auc as _auc

from scanpath.utils.config import load_config, get_cli_args
from scanpath.deepgaze_engine import (
    preprocess_to_inmemory,
    preprocess_to_resized_filestimuli,
    make_spatial_model,
    make_scanpath_model,
    make_spatial_loader,
    make_scanpath_loader,
    make_scheduler,
    convert_fixation_trains,
    convert_stimuli,
)
from scanpath.data.parquet_to_pysaliency import parquet_to_pysaliency


# ===========================================================================
# Shared helpers
# ===========================================================================

class _ZeroCenterbias:
    """Flat (uniform) centerbias — avoids disk I/O, fine for synthetic data."""
    def log_density(self, stimulus):
        return np.zeros((stimulus.shape[0], stimulus.shape[1]))

    def information_gain(self, stimuli, fixations, **kwargs):
        return 0.0


def _make_centerbias(cb_cfg, train_stimuli, train_fixations):
    """Build the appropriate centerbias model from the config centerbias block."""
    if not cb_cfg.use_centerbias:
        return _ZeroCenterbias(), 0.0, None   # centerbias, train_ll placeholder, val_ll placeholder
    if cb_cfg.crossvalidated:
        model = CrossvalidatedBaselineModel(
            train_stimuli, train_fixations,
            bandwidth=cb_cfg.bandwidth, eps=cb_cfg.eps, caching=False,
        )
    else:
        model = BaselineModel(
            stimuli=train_stimuli, fixations=train_fixations,
            bandwidth=cb_cfg.bandwidth, eps=cb_cfg.eps, caching=False,
        )
    return model, None, None   # ll computed later by caller


class TrainOnlyCenterbias(pysaliency.Model):
    """DG3's KDE center-bias prior, estimated from the TRAINING split only
    (fixed-split protocol, decided 2026-09-03).

    Training stimuli get the leave-one-image-out density exactly as in the
    DG3 cross-validation protocol; any other stimulus (validation / test
    images) gets the density of all training fixations, so held-out gaze
    data never enters the prior the model receives as input. Bandwidth and
    eps are the DG3 MIT1003 values from the config.
    """

    def __init__(self, train_stimuli, train_fixations, bandwidth, eps):
        super().__init__(caching=False)
        self.loo = CrossvalidatedBaselineModel(
            train_stimuli, train_fixations, bandwidth=bandwidth, eps=eps, caching=False)
        self.full = BaselineModel(
            stimuli=train_stimuli, fixations=train_fixations,
            bandwidth=bandwidth, eps=eps, caching=False)
        self._train_ids = set(train_stimuli.stimulus_ids)

    def _log_density(self, stimulus):
        from pysaliency.datasets import get_image_hash
        if get_image_hash(stimulus) in self._train_ids:
            return self.loo._log_density(stimulus)
        return self.full._log_density(stimulus)


def _lmdb(use_lmdb: bool, base_dir: Path, name: str, lmdb_base_dir=None):
    """Return an LMDB path string if use_lmdb is enabled, else None.

    If *lmdb_base_dir* is given, LMDB files are written there (e.g. local
    scratch on the GPU node) instead of inside *base_dir*. This avoids
    Lustre I/O for the hot image cache and leads to much faster epochs.
    """
    if not use_lmdb:
        return None
    root = Path(lmdb_base_dir) if lmdb_base_dir else base_dir
    p = root / 'lmdb_cache' / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _build_opt_sched(model, lr: float, milestones):
    """Return an (Adam optimizer, MultiStepLR scheduler) pair."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = make_scheduler(optimizer, milestones)
    return optimizer, scheduler


def _freeze_scopes(model, scopes):
    """Freeze all parameters whose names start with any entry in *scopes*."""
    for scope in scopes:
        for param_name, param in model.named_parameters():
            if param_name.startswith(scope):
                print(f"  Freezing: {param_name}")
                param.requires_grad = False


def _print_param_stats(model, label: str = ""):
    """Print trainable / frozen parameter counts per sub-network and in total."""
    sub_names = ["features", "saliency_network", "scanpath_network", "fixation_selection_network"]
    header = f"  {'sub-network':<42}  {'trainable':>12}  {'frozen':>12}  {'total':>12}"
    sep    = "  " + "-" * (len(header) - 2)
    prefix = f"\n  Parameter summary{f' — {label}' if label else ''}:"
    print(prefix)
    print(header)
    print(sep)
    grand_train = grand_frozen = 0
    for name in sub_names:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        n_train  = sum(p.numel() for p in sub.parameters() if     p.requires_grad)
        n_frozen = sum(p.numel() for p in sub.parameters() if not p.requires_grad)
        grand_train  += n_train
        grand_frozen += n_frozen
        print(f"  {name:<42}  {n_train:>12,}  {n_frozen:>12,}  {n_train+n_frozen:>12,}")
    print(sep)
    print(f"  {'TOTAL':<42}  {grand_train:>12,}  {grand_frozen:>12,}  {grand_train+grand_frozen:>12,}")
    print()


def _resolve_ckpt(stage_dir: Path) -> Path:
    """Return final.pth if it exists, fall back to best.pth. Raise if neither exists."""
    for name in ('final.pth', 'best.pth'):
        p = stage_dir / name
        if p.exists():
            if name == 'best.pth':
                print(f"Warning: final.pth not found, falling back to best.pth in {stage_dir}")
            return p
    raise FileNotFoundError(
        f"No checkpoint (final.pth / best.pth) found in {stage_dir}"
    )


def _make_run_name(cfg):
    """Return a wandb run name.

    If utils.run_name is set in the config, use it verbatim.
    Otherwise generate a datetime+hash name identical to the autoreg convention:
        2026_03_03_14_32_11_a3f8c21b
    """
    name = getattr(cfg.utils, 'run_name', None)
    if name:
        return str(name)
    date_time  = time.strftime("%Y_%m_%d_%H_%M_%S")
    short_hash = uuid.uuid4().hex[:8]
    return f"{date_time}_{short_hash}"


# ===========================================================================
# Mixed-precision (bf16) support
# ===========================================================================

def _apply_mixed_precision():
    """Monkeypatch deepgaze_pytorch.training with bf16-autocast epoch functions.

    Why monkeypatching instead of editing the submodule:
      _train() calls train_epoch / eval_epoch by name in its module namespace,
      so replacing them on the module object is enough — no submodule edits needed.

    Why bf16 and not fp16:
      bf16 has the same exponent range as fp32, so gradients never underflow and
      GradScaler is not required.  On A100, bf16 throughput matches fp16.

    Why cast log_density to float32 after the forward pass in eval:
      The AUC metric calls .cpu().numpy() on tensor values; numpy has no bf16 dtype
      and would raise a TypeError.  log_likelihood and nss are pure tensor ops and
      work fine in bf16, but we cast all metrics for consistency.
    """
    from tqdm import tqdm as _tqdm

    def _amp_train_epoch(model, dataset, optimizer, device):
        model.train()
        losses = []
        batch_weights = []
        pbar = _tqdm(dataset)
        for batch in pbar:
            optimizer.zero_grad()
            image         = batch.pop('image').to(device)
            centerbias    = batch.pop('centerbias').to(device)
            fixation_mask = batch.pop('fixation_mask').to(device)
            x_hist        = batch.pop('x_hist', torch.tensor([])).to(device)
            y_hist        = batch.pop('y_hist', torch.tensor([])).to(device)
            weights       = batch.pop('weight').to(device)
            durations     = batch.pop('durations', torch.tensor([])).to(device)
            kwargs        = {k: v.to(device) for k, v in dict(batch).items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if isinstance(model, _DeepGazeII):
                    log_density = model(image, centerbias, **kwargs)
                else:
                    log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)
                loss = -_log_likelihood(log_density, fixation_mask, weights=weights)

            losses.append(loss.detach().float().cpu().numpy())
            batch_weights.append(weights.detach().cpu().numpy().sum())
            pbar.set_description('{:.05f}'.format(np.average(losses, weights=batch_weights)))
            loss.backward()
            optimizer.step()
        return np.average(losses, weights=batch_weights)

    def _amp_eval_epoch(model, dataset, baseline_information_gain, device, metrics=None):
        model.eval()
        if metrics is None:
            metrics = ['LL', 'IG', 'NSS', 'AUC']
        metric_functions = {'LL': _log_likelihood, 'NSS': _nss, 'AUC': _auc}
        metric_scores = {}
        batch_weights = []
        with torch.no_grad():
            pbar = _tqdm(dataset)
            for batch in pbar:
                image         = batch.pop('image').to(device)
                centerbias    = batch.pop('centerbias').to(device)
                fixation_mask = batch.pop('fixation_mask').to(device)
                x_hist        = batch.pop('x_hist', torch.tensor([])).to(device)
                y_hist        = batch.pop('y_hist', torch.tensor([])).to(device)
                weights       = batch.pop('weight').to(device)
                durations     = batch.pop('durations', torch.tensor([])).to(device)
                kwargs        = {k: v.to(device) for k, v in dict(batch).items()}

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if isinstance(model, _DeepGazeII):
                        log_density = model(image, centerbias, **kwargs)
                    else:
                        log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)

                # Cast to float32: numpy (used inside auc) has no bf16 dtype.
                log_density = log_density.float()

                for metric_name, metric_fn in metric_functions.items():
                    if metric_name not in metrics:
                        continue
                    metric_scores.setdefault(metric_name, []).append(
                        metric_fn(log_density, fixation_mask, weights=weights).detach().cpu().numpy()
                    )
                batch_weights.append(weights.detach().cpu().numpy().sum())
                for display_metric in ['LL', 'NSS', 'AUC']:
                    if display_metric in metrics:
                        pbar.set_description('{} {:.05f}'.format(
                            display_metric,
                            np.average(metric_scores[display_metric], weights=batch_weights),
                        ))
                        break

        data = {k: np.average(v, weights=batch_weights) for k, v in metric_scores.items()}
        if 'IG' in metrics:
            data['IG'] = data['LL'] - baseline_information_gain
        return data

    _dg_training.train_epoch = _amp_train_epoch
    _dg_training.eval_epoch  = _amp_eval_epoch
    print("Mixed precision (bf16) enabled — patched train_epoch and eval_epoch.")


# ===========================================================================
# imagenet mode — two-stage training on synthetic scanpaths
# ===========================================================================

def run_imagenet(cfg, run_name):
    import math
    t = cfg.training
    device = cfg.model.device
    base_dir = Path(t.out_dir)   # e.g. runs/imagenet_paths

    # --- Data ---
    print("Loading parquet data …")
    train_stimuli, train_fixations, val_stimuli, val_fixations = parquet_to_pysaliency(
        parquet_path=t.parquet_path,
        train_frac=t.train_frac,
        seed=t.seed,
        image_base_dir=t.image_base_dir,
        target_size=t.image_size,
        train_n_images=t.train_n_images,
        val_n_images=getattr(t, 'val_n_images', None),
        train_n_scanpaths=getattr(t, 'train_n_scanpaths', None),
        val_n_scanpaths=getattr(t, 'val_n_scanpaths', None),
    )

    centerbias, _, _ = _make_centerbias(t.centerbias, train_stimuli, train_fixations)

    in_memory = getattr(t, 'in_memory', True)
    if in_memory:
        print("Pre-loading and resizing images into RAM …")
        train_stimuli = preprocess_to_inmemory(train_stimuli)
        val_stimuli   = preprocess_to_inmemory(val_stimuli)
    else:
        print("Disk-read mode: images will be lazily resized per batch (low RAM).")
        train_stimuli = preprocess_to_resized_filestimuli(train_stimuli)
        val_stimuli   = preprocess_to_resized_filestimuli(val_stimuli)

    # Default output dirs live under a unique run_name folder — always fresh.
    # Override with stage1.path / stage2.path for explicit control or reuse.
    run_dir      = base_dir / run_name
    spatial_dir  = Path(t.stage1.path) if t.stage1.path else run_dir / "spatial"
    scanpath_dir = Path(t.stage2.path) if t.stage2.path else run_dir / "scanpath"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    scanpath_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage 1 dir: {spatial_dir}")
    print(f"Stage 2 dir: {scanpath_dir}")

    # --- Derive max_epochs from nr_of_images_seen if set (per stage) ---
    n_train = len(train_fixations)
    for stage_cfg in [t.stage1, t.stage2]:
        n_seen = getattr(stage_cfg, 'nr_of_images_seen', None)
        if n_seen:
            stage_cfg.max_epochs = math.ceil(n_seen / n_train)

    batches_per_epoch = math.ceil(n_train / t.batch_size)
    print(
        f"\n  Training plan:"
        f"\n    n_train_fixations : {n_train}"
        f"\n    batch_size        : {t.batch_size}"
        f"\n    batches/epoch     : {batches_per_epoch}"
        f"\n    stage1 max_epochs : {t.stage1.max_epochs}  "
        f"({t.stage1.max_epochs * n_train:,} fixations total)"
        f"\n    stage2 max_epochs : {t.stage2.max_epochs}  "
        f"({t.stage2.max_epochs * n_train:,} fixations total)\n"
    )

    # Write config snapshot into the Stage 2 dir (unique per run).
    with open(scanpath_dir / 'config.json', 'w') as _f:
        json.dump({'run_name': run_name, 'config': cfg.toDict()}, _f, indent=2, default=str)

    spatial_ckpt = spatial_dir / 'final.pth'

    # --- Stage 1: spatial saliency model ---
    s1_steps = 0
    if t.stage1.use_pretrained and spatial_ckpt.exists():
        print(f"\n=== Stage 1: REUSING pretrained spatial model ===\n  {spatial_ckpt}")
    else:
        print(f"\n=== Stage 1: spatial model ===")
        model              = make_spatial_model(cfg, device)
        _print_param_stats(model, "Stage 1 — spatial model")
        optimizer, scheduler = _build_opt_sched(
            model, t.stage1.lr or t.optimizer.lr, t.stage1.lr_milestones)

        t_loader = make_spatial_loader(
            train_stimuli, train_fixations, centerbias,
            t.batch_size, t.num_workers, cached=in_memory,
        )
        v_loader = make_spatial_loader(
            val_stimuli, val_fixations, centerbias,
            t.batch_size, t.num_workers, cached=in_memory,
        )

        s1_steps = _train(
            spatial_dir, model, t_loader, 0.0, v_loader, 0.0,
            optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
            max_epochs=t.stage1.max_epochs,
            stage_name="stage1",
        ) or 0

    # --- Stage 2: full scanpath model (init from Stage 1) ---
    if t.stage2.use_pretrained and (scanpath_dir / 'final.pth').exists():
        print(f"\n=== Stage 2: REUSING pretrained scanpath model ===\n  {scanpath_dir / 'final.pth'}")
        return

    print(f"\n=== Stage 2: scanpath model ===")
    model                = make_scanpath_model(cfg, "stage2", device)
    _print_param_stats(model, "Stage 2 — scanpath model")
    optimizer, scheduler = _build_opt_sched(
            model, t.stage2.lr or t.optimizer.lr, t.stage2.lr_milestones)

    t_loader = make_scanpath_loader(
        train_stimuli, train_fixations, centerbias,
        t.batch_size, t.num_workers, cache_image_data=in_memory,
    )
    v_loader = make_scanpath_loader(
        val_stimuli, val_fixations, centerbias,
        t.batch_size, t.num_workers, cache_image_data=in_memory,
    )

    _train(
        scanpath_dir, model, t_loader, 0.0, v_loader, 0.0,
        optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
        startwith=_resolve_ckpt(spatial_dir),
        max_epochs=t.stage2.max_epochs,
        stage_name="stage2",
        wandb_step_offset=s1_steps + 1,
    )


# ===========================================================================
# salicon mode — one-stage spatial pretraining
# ===========================================================================

def run_salicon(cfg):
    t = cfg.training
    device = cfg.model.device
    out_dir = Path(t.out_dir)
    dataset_dir = Path(t.dataset_dir)

    # --- Data ---
    print("Loading SALICON …")
    train_stimuli, train_fixations = pysaliency.get_SALICON_train(location=dataset_dir)
    val_stimuli,   val_fixations   = pysaliency.get_SALICON_val(location=dataset_dir)

    centerbias, _, _ = _make_centerbias(t.centerbias, train_stimuli, train_fixations)

    print("Computing baseline log-likelihoods …")
    train_ll = centerbias.information_gain(
        train_stimuli, train_fixations, verbose=True, average='image')
    val_ll = centerbias.information_gain(
        val_stimuli, val_fixations, verbose=True, average='image')

    # Store LMDB in dataset_dir so it is reused across multiple SALICON runs.
    # Override with lmdb_base_dir config (e.g. for scratch or a different disk).
    salicon_lmdb_base = getattr(t, 'lmdb_base_dir', None) or dataset_dir
    t_loader = make_spatial_loader(
        train_stimuli, train_fixations, centerbias,
        t.batch_size, t.num_workers,
        lmdb_path=_lmdb(t.use_lmdb, dataset_dir, 'SALICON_train', lmdb_base_dir=salicon_lmdb_base),
    )
    v_loader = make_spatial_loader(
        val_stimuli, val_fixations, centerbias,
        t.batch_size, t.num_workers,
        lmdb_path=_lmdb(t.use_lmdb, dataset_dir, 'SALICON_val', lmdb_base_dir=salicon_lmdb_base),
    )

    model                = make_spatial_model(cfg, device)
    optimizer, scheduler = _build_opt_sched(model, t.optimizer.lr, t.stage1.lr_milestones)

    print(f"\n=== SALICON pretraining ===")
    _print_param_stats(model, "SALICON spatial pretraining")
    _train(
        out_dir / 'pretraining', model, t_loader, train_ll, v_loader, val_ll,
        optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
        max_epochs=getattr(t, 'max_epochs_pretraining', None),
    )


# ===========================================================================
# mit1003 mode — three-stage cross-validated training
# ===========================================================================

def run_mit1003(cfg, fold_only=None):
    t = cfg.training
    device = cfg.model.device
    out_dir = Path(t.out_dir)
    dataset_dir = Path(t.dataset_dir)

    # --- Locate pretrain checkpoint ---
    # Two modes:
    #   scanpath_pretrain_path — direct path to a full scanpath .pth file.
    #       Stage 1 (spatial-only finetune) is SKIPPED; Stage 2 initialises
    #       directly from this checkpoint (all weights loaded, strict=False).
    #       Use this for imagenet-pretrained scanpath models.
    #   spatial_pretrain_path  — path to a SALICON run directory.
    #       Stage 1 spatially fine-tunes MIT1003 from pretraining/final.pth,
    #       then Stage 2 adds the scanpath head.  Standard SALICON→MIT1003 flow.
    _scanpath_pretrain_path = getattr(t, 'scanpath_pretrain_path', None)
    if _scanpath_pretrain_path:
        scanpath_pretrain_ckpt = Path(_scanpath_pretrain_path)
        if not scanpath_pretrain_ckpt.exists():
            raise FileNotFoundError(
                f"scanpath_pretrain_path not found: {scanpath_pretrain_ckpt}"
            )
        spatial_ckpt = None  # Stage 1 will be skipped
        print(f"Using full scanpath pretrain checkpoint (Stage 1 skipped): {scanpath_pretrain_ckpt}")
    else:
        scanpath_pretrain_ckpt = None
        raw = str(t.spatial_pretrain_path or '').strip()
        if raw.lower() in ('', 'none', 'null', '~'):
            # No spatial pretrain — Stage 1 starts from random init (DenseNet backbone only).
            spatial_ckpt = None
            print("No spatial pretrain checkpoint — Stage 1 starts from random init.")
        elif Path(raw).suffix == '.pth':
            # Direct path to a .pth file (e.g. imagenet spatial checkpoint).
            spatial_ckpt = Path(raw)
            if not spatial_ckpt.exists():
                raise FileNotFoundError(
                    f"spatial_pretrain_path not found: {spatial_ckpt}"
                )
            print(f"Using spatial pretrain checkpoint (direct): {spatial_ckpt}")
        else:
            # Directory with pretraining/final.pth (SALICON run dir).
            spatial_pretrain_raw = Path(raw)
            spatial_ckpt = spatial_pretrain_raw / 'pretraining' / 'final.pth'
            if not spatial_ckpt.exists():
                spatial_ckpt = spatial_pretrain_raw / 'pretraining' / 'best.pth'
            if not spatial_ckpt.exists():
                raise FileNotFoundError(
                    f"No spatial pretrain checkpoint (final.pth / best.pth) found in:\n"
                    f"  {spatial_pretrain_raw / 'pretraining'}\n"
                    "Run SALICON pretraining first:\n"
                    "  python -m scanpath.train_deepgaze --config scanpath/configs/salicon.yaml"
                )
            print(f"Using spatial pretrain checkpoint: {spatial_ckpt}")

    # --- Load and convert MIT1003 to two-size ---
    print("Loading MIT1003 …")
    mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=dataset_dir, replace_initial_invalid_fixations=True
    )

    print("Converting stimuli to two-size (768×1024 / 1024×768) …")
    mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_orig, mit_scanpaths_orig)
    mit_stimuli_twosize   = convert_stimuli(mit_stimuli_orig, out_dir / 'MIT1003_twosize')

    # Drop the initial forced fixation — it's used for conditioning only
    mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.lengths > 0]

    MIT1003_centerbias, _, _ = _make_centerbias(t.centerbias, mit_stimuli_twosize, mit_fixations_twosize)

    # Fixed image-wise splits from text files (one basename per line,
    # train.txt / validation.txt). Replaces cross-validation with a single
    # training run on the given split; every image must resolve to a stimulus.
    fixed_split_dir = getattr(t, 'fixed_split_dir', None)
    if fixed_split_dir:
        fixed_split_dir = Path(fixed_split_dir)
        basenames = [Path(fn).name.lower() for fn in mit_stimuli_twosize.filenames]
        name_to_idx = {bn: i for i, bn in enumerate(basenames)}

        def _load_split(fname):
            wanted = [l.strip() for l in open(fixed_split_dir / fname) if l.strip()]
            missing = [w for w in wanted if Path(w).name.lower() not in name_to_idx]
            if missing:
                raise SystemExit(f"{fname}: {len(missing)} images not found in "
                                 f"MIT1003 stimuli, e.g. {missing[:3]}")
            return sorted(name_to_idx[Path(w).name.lower()] for w in wanted)

        fixed_train_idx = _load_split('train.txt')
        fixed_val_idx = _load_split('validation.txt')
        overlap = set(fixed_train_idx) & set(fixed_val_idx)
        if overlap:
            raise SystemExit(f"train/val overlap: {len(overlap)} images")
        print(f"MIT1003 FIXED splits from {fixed_split_dir}: "
              f"{len(fixed_train_idx)} train / {len(fixed_val_idx)} val images")
        if t.centerbias.use_centerbias:
            # Fixed-split protocol (2026-09-03): the center-bias prior the
            # model receives as input is estimated from the training split
            # only (the all-image LOO prior above is the CV-era formulation).
            _cb_stim, _cb_fix = pysaliency.create_subset(
                mit_stimuli_twosize, mit_fixations_twosize, fixed_train_idx)
            _cb_fix.n = _cb_fix.n.astype(int)
            MIT1003_centerbias = TrainOnlyCenterbias(
                _cb_stim, _cb_fix, t.centerbias.bandwidth, t.centerbias.eps)
            print(f"center bias: TRAIN-ONLY prior ({len(fixed_train_idx)} images; "
                  "LOO on training stimuli, full train KDE elsewhere)")

    crossval_folds = t.crossval_folds
    if fixed_split_dir:
        pass
    elif crossval_folds < 2:
        raise ValueError(
            f"crossval_folds must be >= 2 (got {crossval_folds}).\n"
            "With 1 fold, pysaliency's train_split returns 0 fixations.\n"
            "Use crossval_folds >= 5 (pysaliency's 2-fold split is also broken "
            "with plain Fixations input; use configs/mit1003_smoketest.yaml for quick tests)."
        )
    if fixed_split_dir:
        folds_to_run = [0]
    elif fold_only is not None:
        if not (0 <= fold_only < crossval_folds):
            raise ValueError(f"--fold_only {fold_only} is out of range for crossval_folds={crossval_folds}")
        folds_to_run = [fold_only]
        print(f"Running single fold: {fold_only} / {crossval_folds}")
    else:
        folds_to_run = range(crossval_folds)
    frozen_scopes  = list(t.stage2.frozen_scopes)

    lmdb_base = getattr(t, 'lmdb_base_dir', None) or None

    def _fold_lmdb(name: str):
        return _lmdb(t.use_lmdb, out_dir, name, lmdb_base_dir=lmdb_base)

    # --- Cross-validation loop ---
    # WandB requires the `step` arg to wandb.log() to be monotonically
    # non-decreasing within a single run — it is a run-level global counter,
    # not per-metric.  Without this accumulator, every fold would reset the
    # offset to 0 and WandB would silently drop all logs from fold 1 onward.
    wandb_total_steps = 0

    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"=== Cross-validation fold {fold + 1} / {crossval_folds} ===")
        print(f"{'='*60}")

        fold_split_seed = getattr(t, 'fold_split_seed', None)
        if fixed_split_dir:
            train_stimuli, train_fixations = pysaliency.create_subset(
                mit_stimuli_twosize, mit_fixations_twosize, fixed_train_idx)
            val_stimuli, val_fixations = pysaliency.create_subset(
                mit_stimuli_twosize, mit_fixations_twosize, fixed_val_idx)
        elif fold_split_seed is not None:
            # Image-wise split identical to fixation-prediction's train.py /
            # eval_mit1003_fair.py (seed 42 there), for cross-model fold parity:
            #   perm = default_rng(seed).permutation(n); array_split; fold k = val.
            # Stimulus order matches: both sides read pysaliency's MIT1003 cache.
            n_img = len(mit_stimuli_twosize.filenames)
            perm = np.random.default_rng(int(fold_split_seed)).permutation(n_img)
            split = np.array_split(perm, crossval_folds)
            val_idx = sorted(split[fold].tolist())
            train_idx = sorted(np.concatenate(
                [split[i] for i in range(crossval_folds) if i != fold]).tolist())
            print(f"fold_split_seed={fold_split_seed}: ours split, "
                  f"{len(train_idx)} train / {len(val_idx)} val images")
            train_stimuli, train_fixations = pysaliency.create_subset(
                mit_stimuli_twosize, mit_fixations_twosize, train_idx)
            val_stimuli, val_fixations = pysaliency.create_subset(
                mit_stimuli_twosize, mit_fixations_twosize, val_idx)
        else:
            train_stimuli, train_fixations = pysaliency.dataset_config.train_split(
                mit_stimuli_twosize, mit_fixations_twosize,
                crossval_folds=crossval_folds, fold_no=fold,
            )
            val_stimuli, val_fixations = pysaliency.dataset_config.validation_split(
                mit_stimuli_twosize, mit_fixations_twosize,
                crossval_folds=crossval_folds, fold_no=fold,
            )
        # pysaliency can return float-dtype .n after reindexing; guard against that.
        train_fixations.n = train_fixations.n.astype(int)
        val_fixations.n   = val_fixations.n.astype(int)

        train_ll = MIT1003_centerbias.information_gain(
            train_stimuli, train_fixations, verbose=True, average='image')
        val_ll = MIT1003_centerbias.information_gain(
            val_stimuli, val_fixations, verbose=True, average='image')

        tag = 'fixedsplit' if fixed_split_dir else f'crossval-{crossval_folds}-{fold}'

        # -- Stage 1: spatial finetune on MIT1003 --
        spatial_fold_dir = out_dir / 'MIT1003_spatial' / tag
        if scanpath_pretrain_ckpt:
            # Full scanpath checkpoint provided — skip the spatial-only stage.
            # The imagenet scanpath weights are loaded directly into Stage 2.
            print(f"\n--- Fold {fold}: Stage 1 — SKIPPED (scanpath_pretrain_path set) ---")
            s1_steps = 0
        else:
            # MIT1003 is small (~1003 images); load everything into CPU RAM once.
            print(f"\n--- Fold {fold}: Stage 1 — spatial finetune ---")
            t_spatial = make_spatial_loader(
                train_stimuli, train_fixations, MIT1003_centerbias,
                t.batch_size, t.num_workers,
                cached=True,
            )
            v_spatial = make_spatial_loader(
                val_stimuli, val_fixations, MIT1003_centerbias,
                t.batch_size, t.num_workers,
                cached=True,
            )

            model                = make_spatial_model(cfg, device)
            optimizer, scheduler = _build_opt_sched(
                model, t.stage1.lr or t.optimizer.lr, t.stage1.lr_milestones
            )
            if fold == 0:
                _print_param_stats(model, f"Stage 1 — spatial finetune (fold {fold})")

            s1_steps = _train(
                spatial_fold_dir, model, t_spatial, train_ll, v_spatial, val_ll,
                optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
                startwith=spatial_ckpt,
                stage_name=f'stage1-{tag}',
                max_epochs=getattr(t.stage1, 'max_epochs', None),
                wandb_step_offset=wandb_total_steps,
            ) or 0

            # Free stage 1 objects before allocating stage 2
            del t_spatial, v_spatial, model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

        wandb_total_steps += s1_steps + 1

        # -- Stage 2: partially frozen scanpath --
        print(f"\n--- Fold {fold}: Stage 2 — partially frozen scanpath ---")
        t_scanpath = make_scanpath_loader(
            train_stimuli, train_fixations, MIT1003_centerbias,
            t.batch_size, t.num_workers,
            cache_image_data=True,
        )
        v_scanpath = make_scanpath_loader(
            val_stimuli, val_fixations, MIT1003_centerbias,
            t.batch_size, t.num_workers,
            cache_image_data=True,
        )

        frozen_dir           = out_dir / 'MIT1003_scanpath_partially_frozen_saliency_network' / tag
        model                = make_scanpath_model(cfg, "stage2", device)
        if t.stage2.use_frozen_scopes:
            _freeze_scopes(model, frozen_scopes)
        if fold == 0:
            _print_param_stats(model, f"Stage 2 — partially frozen scanpath (fold {fold})")
        optimizer, scheduler = _build_opt_sched(
            model, t.stage2.lr or t.optimizer.lr, t.stage2.lr_milestones
        )

        _module_ckpt = getattr(t, 'scanpath_module_pretrain_path', None)
        if _module_ckpt and str(_module_ckpt).lower() not in ('', 'none', 'null', '~'):
            if scanpath_pretrain_ckpt:
                raise ValueError("scanpath_module_pretrain_path and scanpath_pretrain_path "
                                 "are mutually exclusive")
            if not (frozen_dir / 'last.pth').exists():
                # Scanpath-module transfer (2026-09-03): pre-load the FULL
                # synthetic scanpath checkpoint, then let _train() load the
                # MIT1003 stage-1 checkpoint over it (strict=False). MIT wins
                # on every overlapping key (saliency net, spatial part of the
                # fixation-selection net, finalizer); the synthetic
                # scanpath_network.* and the history-specific *part1* keys,
                # absent from stage 1, survive. Identical stages 1-3 to the
                # other arms; unlike scanpath_pretrain_path, stage 1 is NOT
                # skipped (engineering_log §18).
                print(f"scanpath-module transfer: pre-loading {_module_ckpt} "
                      f"before the stage-1 overlay")
                _dg_training.load_model_weights(model, str(_module_ckpt))

        s2_steps = _train(
            frozen_dir, model, t_scanpath, train_ll, v_scanpath, val_ll,
            optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
            startwith=scanpath_pretrain_ckpt if scanpath_pretrain_ckpt else _resolve_ckpt(spatial_fold_dir),
            stage_name=f'stage2-{tag}',
            max_epochs=getattr(t.stage2, 'max_epochs', None),
            wandb_step_offset=wandb_total_steps,
        ) or 0
        wandb_total_steps += s2_steps + 1

        # -- Stage 3: full model finetune --
        print(f"\n--- Fold {fold}: Stage 3 — full scanpath finetune ---")
        model                = make_scanpath_model(cfg, "stage3", device)
        if fold == 0:
            _print_param_stats(model, f"Stage 3 — full finetune (fold {fold})")
        optimizer, scheduler = _build_opt_sched(
            model, t.stage3.lr or t.optimizer.lr, t.stage3.lr_milestones
        )

        # t_scanpath / v_scanpath reused from Stage 2 (same in-RAM data)
        s3_steps = _train(
            out_dir / 'MIT1003_scanpath' / tag,
            model, t_scanpath, train_ll, v_scanpath, val_ll,
            optimizer, scheduler, minimum_learning_rate=1e-7, device=device,
            startwith=_resolve_ckpt(frozen_dir),
            stage_name=f'stage3-{tag}',
            max_epochs=getattr(t.stage3, 'max_epochs', None),
            wandb_step_offset=wandb_total_steps,
        ) or 0
        wandb_total_steps += s3_steps + 1

        # Free all fold objects before next iteration
        del t_scanpath, v_scanpath, model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    cli = get_cli_args()
    config_path = cli.pop("config") or str(
        Path(__file__).parent / "configs" / "imagenet.yaml"
    )
    # These are script-level args, not config leaves — extract before load_config.
    fold_only   = cli.pop("fold_only",   None)
    wandb_group = cli.pop("wandb_group", None)
    cfg = load_config(config_path, cli)

    if getattr(cfg.training, 'mixed_precision', False):
        _apply_mixed_precision()

    # Switch DataLoader IPC from /dev/shm (shared memory) to file-system.
    # /dev/shm is tiny on SLURM nodes; with many workers + large batches it
    # causes malloc corruption / worker segfaults.
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Suppress per-batch tqdm bars — they clutter SLURM logs without adding
    # information beyond what the per-epoch CSV/print already captures.
    os.environ["TQDM_DISABLE"] = "1"

    # Silence all wandb output in SLURM logs — our CSV + prints are the source
    # of truth; wandb is used only for the web dashboard.
    os.environ["WANDB_SILENT"]  = "true"   # suppresses startup banner
    import logging
    logging.getLogger("wandb").setLevel(logging.ERROR)

    # Suppress noisy pysaliency UserWarning about FixationTrains attributes.
    warnings.filterwarnings("ignore", message="don't use attributes for FixationTrains")
    # Suppress PyTorch sparse tensor deprecation warning from DeepGaze data.py.
    warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor")
    # Suppress cuDNN v8 execution-plan fallback warning — PyTorch retries
    # automatically with a supported plan; the warning is purely informational.
    warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")

    # Optional W&B initialisation
    run_name = _make_run_name(cfg)
    print(f"Run name: {run_name}")

    # Save fully-resolved config + run metadata to the output directory.
    # This mirrors exactly what is logged to W&B so results are reproducible
    # even without a W&B connection.
    out_dir = Path(cfg.training.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = out_dir / 'config.json'
    config_snapshot = {
        'run_name':   run_name,
        'config_file': str(Path(config_path).resolve()),
        'config':     cfg.toDict(),
    }
    with open(config_save_path, 'w') as _f:
        json.dump(config_snapshot, _f, indent=2, default=str)
    print(f"Config saved to: {config_save_path}")

    if cfg.utils.use_wandb:
        wandb_kwargs = dict(
            project=cfg.utils.wandb_project,
            name=run_name,
        )
        if cfg.utils.wandb_dir:
            wandb_kwargs["dir"] = cfg.utils.wandb_dir
        if wandb_group:
            wandb_kwargs["group"] = wandb_group
        wandb.init(**wandb_kwargs, config=dict(cfg.training))
        print(f"W&B run: {wandb.run.url}")

    mode = cfg.training.mode
    if mode == "imagenet":
        run_imagenet(cfg, run_name=run_name)
    elif mode == "salicon":
        run_salicon(cfg)
    elif mode == "mit1003":
        run_mit1003(cfg, fold_only=fold_only)
    else:
        raise ValueError(
            f"Unknown training.mode: '{mode}'. Expected: imagenet | salicon | mit1003"
        )


if __name__ == "__main__":
    main()
