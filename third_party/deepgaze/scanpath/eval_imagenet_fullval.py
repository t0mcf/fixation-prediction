"""Evaluate DG3 ImageNet-synthetic checkpoints on the full canonical validation set.

Protocol: same 5000 canonical val images as scanpather/fixation-prediction,
no scanpath cap, so 80,000 scanpaths and 1,200,000 predicted fixations.

With --our-metrics the same predictions are additionally scored with the metric
implementations from the fixation-prediction repo, on a 64x64 grid. This exists
because DG3's own nss() unpacks torch.std_mean as (mean, std) -- the two are
returned the other way round -- so it computes (p - std) / mean rather than
(p - mean) / std. Since mean is 1/N for a normalised density, that value scales
with the pixel count and is not NSS. DG3's log_likelihood() and auc() are fine;
only NSS is affected.

Grid choice: our model predicts natively on 64x64, DG3 on 224x224. Densities are
pooled down rather than ours interpolated up, because area-pooling a smooth map
loses little, whereas upsampling a coarse one manufactures sharpness that NSS
then rewards. Both models are therefore scored on the grid the coarser one
actually predicts on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make models/DeepGaze importable without installing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "DeepGaze"))

from deepgaze_pytorch.metrics import log_likelihood as _log_likelihood, nss as _nss, auc as _auc
from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
from deepgaze_pytorch.features.densenet import RGBDenseNet201

from scanpath.data.parquet_to_pysaliency import parquet_to_pysaliency
from scanpath.deepgaze_engine import (
    make_scanpath_loader,
    preprocess_to_resized_filestimuli,
    build_saliency_network,
    build_scanpath_network,
    build_fixation_selection_network,
)
from scanpath.train_deepgaze import _ZeroCenterbias


DEFAULT_PARQUET = "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
DEFAULT_IMAGE_BASE = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images"
DEFAULT_LAYERS = [
    "1.features.denseblock4.denselayer32.norm1",
    "1.features.denseblock4.denselayer32.conv1",
    "1.features.denseblock4.denselayer31.conv2",
]


def _cfg_value(config_path: Path | None, path: list[str], default):
    if config_path is None or not config_path.exists():
        return default
    data = json.loads(config_path.read_text())
    cur = data.get("config", data)
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_model(checkpoint_path: Path, device: str, config_path: Path | None = None) -> DeepGazeIII:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    densenet_layers = _cfg_value(config_path, ["model", "densenet_layers"], DEFAULT_LAYERS)
    downsample = _cfg_value(config_path, ["training", "stage2", "downsample"], 1)
    readout_factor = _cfg_value(config_path, ["training", "stage2", "readout_factor"], 2.25)
    saliency_map_factor = _cfg_value(config_path, ["training", "stage2", "saliency_map_factor"], 1)

    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), list(densenet_layers)),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=downsample,
        readout_factor=readout_factor,
        saliency_map_factor=saliency_map_factor,
        included_fixations=[-1, -2, -3, -4],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


FIXPRED_ROOT = Path("/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/fixation-prediction")


def _load_our_metrics():
    """Import the fixation-prediction metric implementations, or explain why not."""
    if str(FIXPRED_ROOT) not in sys.path:
        sys.path.insert(0, str(FIXPRED_ROOT))
    try:
        from src.training.metrics import nss_score, auc_score
        from src.training.losses import ll_score
    except ImportError as e:
        raise SystemExit(f"--our-metrics needs {FIXPRED_ROOT} importable: {e}")
    return nss_score, auc_score, ll_score


def _to_grid(log_density: torch.Tensor, size: int) -> torch.Tensor:
    """Pool a log-density down to size x size, preserving probability mass."""
    import torch.nn.functional as F
    p = log_density.exp().unsqueeze(1)
    p = F.interpolate(p, size=(size, size), mode="area").squeeze(1)
    p = p / p.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(p.clamp_min(1e-12))


def _mask_to_xy(fixation_mask: torch.Tensor) -> torch.Tensor:
    """Binary mask with one hot pixel per sample -> (B, 2) coordinates in [-1, 1]."""
    B, H, W = fixation_mask.shape
    flat = fixation_mask.reshape(B, -1).argmax(dim=1)
    row, col = flat // W, flat % W
    x = col.float() / (W - 1) * 2.0 - 1.0
    y = row.float() / (H - 1) * 2.0 - 1.0
    return torch.stack([x, y], dim=1)


@torch.no_grad()
def evaluate_on_loader(model, loader, device: str, use_amp: bool,
                       our_metrics: bool = False,
                       our_grid: int = 64) -> dict[str, float]:
    nss_scores, auc_scores, ll_scores, weights_all = [], [], [], []
    our_nss, our_auc, our_ll, our_w = [], [], [], []
    device_type = device.split(":")[0]
    if our_metrics:
        nss_fn, auc_fn, ll_fn = _load_our_metrics()

    for batch_idx, batch in enumerate(loader):
        image = batch.pop("image").to(device)
        centerbias = batch.pop("centerbias").to(device)
        fixation_mask = batch.pop("fixation_mask").to(device)
        x_hist = batch.pop("x_hist", torch.tensor([])).to(device)
        y_hist = batch.pop("y_hist", torch.tensor([])).to(device)
        weights = batch.pop("weight").to(device)
        durations = batch.pop("durations", torch.tensor([])).to(device)
        kwargs = {k: v.to(device) for k, v in dict(batch).items()}

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(use_amp and device_type == "cuda")):
            log_density = model(
                image, centerbias,
                x_hist=x_hist, y_hist=y_hist, durations=durations,
                **kwargs,
            )
        log_density = log_density.float()

        w = weights.detach().cpu().numpy().sum()
        nss_scores.append(_nss(log_density, fixation_mask, weights=weights).detach().cpu().item())
        auc_scores.append(_auc(log_density, fixation_mask, weights=weights).detach().cpu().item())
        ll_scores.append(_log_likelihood(log_density, fixation_mask, weights=weights).detach().cpu().item())
        weights_all.append(w)

        if our_metrics:
            # flat mean over fixations, matching how our own model is scored in
            # fixation-prediction's evaluate(); DG3's per-sample weights are not
            # applied here so the two sides aggregate the same way
            lp = _to_grid(log_density, our_grid)
            xy = _mask_to_xy(fixation_mask).to(lp.device)
            n = lp.shape[0]
            our_nss.append(nss_fn(lp, xy).item())
            our_auc.append(auc_fn(lp, xy).item())
            our_ll.append(ll_fn(lp, xy).item())
            our_w.append(n)

        if batch_idx % 20 == 0:
            ww = np.array(weights_all)
            print(
                f"batch {batch_idx:04d} | samples={int(ww.sum())} | "
                f"NSS={np.average(nss_scores, weights=ww):.4f} | "
                f"AUC={np.average(auc_scores, weights=ww):.4f} | "
                f"LL={np.average(ll_scores, weights=ww):.4f}",
                flush=True,
            )

    w = np.array(weights_all)
    out = {
        "samples": int(w.sum()),
        "NSS": float(np.average(nss_scores, weights=w)),
        "AUC": float(np.average(auc_scores, weights=w)),
        "LL": float(np.average(ll_scores, weights=w)),
    }
    if our_metrics and our_w:
        ow = np.array(our_w)
        out.update({
            "our_grid": our_grid,
            "our_fixations": int(ow.sum()),
            "our_NSS": float(np.average(our_nss, weights=ow)),
            "our_AUC": float(np.average(our_auc, weights=ow)),
            "our_LL": float(np.average(our_ll, weights=ow)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--parquet-path", default=DEFAULT_PARQUET)
    ap.add_argument("--image-base-dir", default=DEFAULT_IMAGE_BASE)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3141)
    ap.add_argument("--val-n-images", type=int, default=5000)
    ap.add_argument("--val-image-list", default=None,
                    help="protocol split file (class/file.JPEG per line) replacing "
                         "the seed-based canonical selection. final evaluation only for "
                         "v2 test_10k.txt.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--our-metrics", action="store_true",
                    help="additionally score with fixation-prediction's metric "
                         "implementations on a common grid (see module docstring)")
    ap.add_argument("--our-grid", type=int, default=64,
                    help="grid size for --our-metrics; 64 = our model's native resolution")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("args:", vars(args), flush=True)

    print("Loading full canonical val data...", flush=True)
    _, _, val_stimuli, val_fixations = parquet_to_pysaliency(
        parquet_path=Path(args.parquet_path),
        seed=args.seed,
        image_base_dir=Path(args.image_base_dir),
        target_size=224,
        train_n_images=1,
        val_n_images=args.val_n_images,
        train_n_scanpaths=1,
        val_n_scanpaths=None,
        val_image_list=Path(args.val_image_list) if args.val_image_list else None,
    )
    val_stimuli = preprocess_to_resized_filestimuli(val_stimuli)
    print(f"val images: {len(val_stimuli)}", flush=True)
    print(f"val fixation targets: {len(val_fixations)}", flush=True)

    loader = make_scanpath_loader(
        val_stimuli, val_fixations, _ZeroCenterbias(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_data=False,
    )
    print(f"val batches: {len(loader)}", flush=True)

    model = load_model(Path(args.checkpoint), device, Path(args.config_json) if args.config_json else None)
    metrics = evaluate_on_loader(model, loader, device, use_amp=not args.no_amp,
                                 our_metrics=args.our_metrics, our_grid=args.our_grid)
    result = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "protocol": "full canonical val, 5000 images, all 16 scanpaths/image",
        "images": len(val_stimuli),
        **metrics,
    }
    print("\nsummary", flush=True)
    for k, v in result.items():
        print(f"{k}: {v}", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
