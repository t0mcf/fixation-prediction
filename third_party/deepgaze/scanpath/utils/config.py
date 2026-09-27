"""
scanpath/utils/config.py
========================
YAML config loading with deep-merge and CLI override support.
Mirrors the pattern from curiosity/autoregressive_detection/autoreg/utils/parser.py.
"""
import argparse
from pathlib import Path
import yaml
from dotmap import DotMap


class StrictDotMap(DotMap):
    """DotMap that raises AttributeError on unknown keys (no silent None returns)."""
    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return super().__getattr__(name)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _selective_update_leaves(cfg: dict, flat: dict) -> dict:
    """Update only leaf nodes of cfg whose key appears in the flat dict."""
    for key, value in cfg.items():
        if isinstance(value, dict):
            _selective_update_leaves(value, flat)
        elif key in flat and flat[key] is not None:
            cfg[key] = flat[key]
    return cfg


def _apply_dotkey_override(cfg: dict, dotkey: str, value) -> bool:
    """Apply a dot-notation CLI override (e.g. 'stage1.lr') to the first matching
    nested path found anywhere in cfg.  Returns True if the key was found."""
    parts = dotkey.split('.')

    def _walk(d: dict, remaining: list) -> bool:
        head, *tail = remaining
        if head in d:
            if not tail:          # reached the leaf
                d[head] = value
                return True
            if isinstance(d[head], dict):
                return _walk(d[head], tail)
        # Not resolved here — search sub-dicts for the same path
        for v in d.values():
            if isinstance(v, dict):
                if _walk(v, remaining):
                    return True
        return False

    return _walk(cfg, parts)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str, relative_to: str) -> str:
    """Resolve path relative to a reference file if it is not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(relative_to).parent / p
    return str(p)


def _load_merged(config_path: str) -> dict:
    """
    Recursively load a yaml file, deep-merging its default_args chain first.
    Each file's default_args are resolved relative to that file's directory,
    so chains like smoketest → mit1003 → base all resolve correctly.
    """
    config_path = str(Path(config_path).resolve())
    cfg = _load_yaml(config_path)
    default_paths = cfg.pop("default_args", [])
    merged: dict = {}
    for path in default_paths:
        resolved = _resolve_path(path, config_path)
        base = _load_merged(resolved)   # recurse
        _merge_dicts(merged, base)
    _merge_dicts(merged, cfg)           # this file wins over its defaults
    return merged


def load_config(task_config_path: str, cli_overrides: dict | None = None) -> StrictDotMap:
    """
    Load config by deep-merging the full default_args chain → CLI overrides.

    default_args chains are resolved recursively, so a smoketest yaml that
    inherits from mit1003.yaml which inherits from base.yaml will see all
    three layers correctly.

    Args:
        task_config_path: Path to the task-specific yaml.
        cli_overrides:    Flat dict of CLI args (None values are ignored).

    Returns:
        StrictDotMap with all config values accessible via dot notation.
    """
    merged = _load_merged(task_config_path)

    # CLI overrides — two mechanisms:
    # • Simple keys (no dot): _selective_update_leaves scans all leaves by name.
    # • Dotted keys (e.g. 'stage1.lr'): _apply_dotkey_override follows the path.
    if cli_overrides:
        simple = {k: v for k, v in cli_overrides.items() if '.' not in k and v is not None}
        dotted = {k: v for k, v in cli_overrides.items() if '.' in k and v is not None}
        if simple:
            _selective_update_leaves(merged, simple)
        for dotkey, value in dotted.items():
            _apply_dotkey_override(merged, dotkey, value)

    return StrictDotMap(merged)


def get_cli_args() -> dict:
    """Parse minimal CLI args. Config path is required; everything else is optional override."""
    parser = argparse.ArgumentParser(description="DeepGaze III training")
    parser.add_argument("--config", type=str, required=False, default=None,
                        help="Path to task config yaml (default: imagenet.yaml)")
    # Generic overrides — add more as needed
    parser.add_argument("--out_dir",        type=str)
    parser.add_argument("--batch_size",     type=int)
    parser.add_argument("--num_workers",    type=int)
    parser.add_argument("--lr",             type=float)
    parser.add_argument("--seed",           type=int)
    parser.add_argument("--device",         type=str)
    parser.add_argument("--crossval_folds",       type=int)
    parser.add_argument("--backbone",             type=str,
                        help="model.backbone: densenet | dinov2 (backbone-swap ablation)")
    parser.add_argument("--fold_split_seed",      type=int,
                        help="Use fixation-prediction's image-wise CV split with this seed "
                             "(42 there) instead of pysaliency's; for cross-model fold parity.")
    parser.add_argument("--train_n_images",         type=int)
    parser.add_argument("--val_n_images",           type=int,
                        help="Fixed val image count. Overrides train_frac split.")
    parser.add_argument("--train_n_scanpaths",      type=int,
                        help="Max scanpaths per training image.")
    parser.add_argument("--val_n_scanpaths",        type=int,
                        help="Max scanpaths per val image.")
    parser.add_argument("--in_memory",             type=_str2bool,
                        help="Pre-load all images into RAM. Set false for large datasets.")
    parser.add_argument("--stage1.lr",              type=float)
    parser.add_argument("--stage1.max_epochs",      type=int)
    parser.add_argument("--stage1.nr_of_images_seen", type=int,
                        help="Total fixations for Stage 1. Overrides stage1.max_epochs.")
    parser.add_argument("--stage1.path",            type=str,
                        help="Save/load path for Stage 1. Null = <out_dir>/<run_name>/spatial/")
    parser.add_argument("--stage1.use_pretrained",  type=_str2bool,
                        help="Skip Stage 1 if stage1.path/final.pth exists.")
    parser.add_argument("--stage2.lr",              type=float)
    parser.add_argument("--stage2.max_epochs",      type=int)
    parser.add_argument("--stage2.nr_of_images_seen", type=int,
                        help="Total fixations for Stage 2. Overrides stage2.max_epochs.")
    parser.add_argument("--stage2.path",            type=str,
                        help="Save path for Stage 2. Null = <out_dir>/<run_name>/scanpath/")
    parser.add_argument("--stage2.use_pretrained",  type=_str2bool,
                        help="Skip Stage 2 if stage2.path/final.pth exists.")
    parser.add_argument("--stage3.max_epochs",          type=int)
    parser.add_argument("--stage3.lr",                  type=float)
    parser.add_argument("--max_epochs_pretraining",     type=int)
    parser.add_argument("--use_wandb",             type=_str2bool)
    parser.add_argument("--wandb_project",         type=str)
    parser.add_argument("--wandb_dir",             type=str,
                        help="Local directory for wandb run files. Overrides base.yaml wandb_dir.")
    parser.add_argument("--run_name",              type=str)
    parser.add_argument("--spatial_pretrain_path",  type=str)
    parser.add_argument("--scanpath_pretrain_path", type=str)
    parser.add_argument("--scanpath_module_pretrain_path", type=str,
                        help="Full synthetic scanpath .pth used to seed the scanpath "
                             "module before the MIT1003 stage-1 overlay (stage 1 still runs).")
    parser.add_argument("--fixed_split_dir",        type=str,
                        help="Directory with train.txt/validation.txt image "
                             "lists; replaces cross-validation with a single "
                             "run on that fixed split.")
    parser.add_argument("--use_frozen_scopes",    type=_str2bool)
    parser.add_argument("--use_lmdb",              type=_str2bool)
    parser.add_argument("--lmdb_base_dir",         type=str)
    parser.add_argument("--readout_factor",          type=float)
    parser.add_argument("--mixed_precision",        type=_str2bool,
                        help="Enable bf16 autocast (A100). No GradScaler needed.")
    parser.add_argument("--fold_only",             type=int,
                        help="Run only this fold index (0-based). Used by array jobs.")
    parser.add_argument("--wandb_group",           type=str,
                        help="WandB group name — set to the shared run name for array jobs.")
    return vars(parser.parse_args())


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
