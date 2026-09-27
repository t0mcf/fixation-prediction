"""
KL / LL / NSS / AUC scores for the autoregressive model that generated the
synthetic scanpaths.

The reported metrics intentionally use the same target construction and metric
functions as src/training/train.py and scripts/evaluate_checkpoint.py. The only
exception is image preprocessing: to replay the generating autoregressive model
faithfully, inputs must use the ImageNet validation transform from the generator
repo (Resize(256) -> CenterCrop(224)), not the DINO model's direct resize.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# -- make autoreg importable without datasetlib --
_dummy = types.ModuleType("datasetlib")
for _n in [
    "ClutteredDigits", "RandomPlaceOnCanvasWithBoxes", "Core50", "Coco2017",
    "COCOSearch18", "ImageNet", "MIT300", "MIT1003", "ImageNetMosaic",
    "COCOSemanticSeg", "ADE20KSemanticSeg", "COCOStuffSeg",
]:
    setattr(_dummy, _n, None)
sys.modules["datasetlib"] = _dummy

AUTOREG_ROOT = "/mnt/vast-nhr/projects/nim00018/tom/autoregressive_detection"
sys.path.insert(0, AUTOREG_ROOT)

from dotmap import DotMap
from autoreg.core.data_and_model_loading import setup_model

# our metrics
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from training.losses import make_gaussian_heatmaps_batch, kl_loss, ll_score
from training.metrics import nss_score, auc_score


META_JSON = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/meta.json"
)
CHECKPOINT = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/model_autoreg_77_classification_64.626.pth"
)
PARQUET = (
    "/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
    "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
)
IMAGENET_ROOT = "/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"
VAL_SEED = 3141
VAL_N = 5000
IMG_SIZE = 224
NUM_GLIMPSES = 15


def _dotmap_deep(d):
    """Recursively convert nested dicts so DotMap leaf values stay as plain types."""
    if isinstance(d, dict):
        return DotMap({k: _dotmap_deep(v) for k, v in d.items()}, _dynamic=False)
    if isinstance(d, list):
        return [_dotmap_deep(v) for v in d]
    return d


def load_generating_model(meta_json: str, checkpoint: str, device: str):
    with open(meta_json) as f:
        meta = json.load(f)
    config = _dotmap_deep(meta["config"])
    config.training.use_pretrained = False
    # spatial_size missing from saved config but needed by neck/postprocessor
    try:
        _ = config.model.config_neck.spatial_size
    except (AttributeError, KeyError):
        config.model.config_neck["spatial_size"] = 64
    model, _ = setup_model(config, img_size=IMG_SIZE)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _canonical_key(path: str) -> str:
    parts = path.split("/")
    return parts[-2] + "/" + parts[-1]


def load_val_data(
    parquet_path: str,
    seed: int,
    n_val: int,
    max_images: int | None = None,
    max_scanpaths_total: int | None = None,
    max_scanpaths_per_image: int | None = None,
    image_list: str | None = None,
) -> dict:
    """Load canonical validation scanpaths grouped by image_path.

    The split and optional caps mirror src.data.dataset.ScanpathDataset:
    select 5000 validation images by canonical key with seed 3141, optionally
    keep the first max_images image paths and the first N scanpaths per image.
    image_list (protocol split file, 'class/file.JPEG' per line) replaces the
    seed-based selection; test_10k.txt is for the final evaluation only.

    Returns {image_path: list of (L, 2) location arrays in [-1, 1]}.
    """
    df = pd.read_parquet(parquet_path, columns=["image_path", "locations"])

    keys = df["image_path"].map(_canonical_key)
    if image_list is not None:
        val_keys = {_canonical_key(l.strip()) for l in open(image_list) if l.strip()}
        missing = val_keys - set(keys.unique())
        if missing:
            raise SystemExit(f"{len(missing)} of {len(val_keys)} images from "
                             f"{image_list} not in the parquet, e.g. {sorted(missing)[:3]}")
        print(f"  explicit image list: {len(val_keys)} images from {image_list}")
    else:
        unique_keys = sorted(keys.unique())
        rng = np.random.default_rng(seed)
        val_keys = set(unique_keys[i] for i in rng.permutation(len(unique_keys))[:n_val])
    df = df[keys.isin(val_keys)].reset_index(drop=True)

    if max_images is not None and max_images > 0:
        kept = df["image_path"].unique()[:max_images]
        df = df[df["image_path"].isin(kept)].reset_index(drop=True)

    # Mirror train.py validation when requested: the training loop uses a
    # sequential val DataLoader and stops after --val-batches batches. For the
    # scaling runs this is 400 * 128 = 51,200 scanpaths.
    if max_scanpaths_total is not None and max_scanpaths_total > 0:
        df = df.iloc[:max_scanpaths_total].reset_index(drop=True)

    if max_scanpaths_per_image is not None and max_scanpaths_per_image > 0:
        df = (
            df.groupby("image_path", sort=False)
            .head(max_scanpaths_per_image)
            .reset_index(drop=True)
        )

    grouped = {}
    for img_path, group in df.groupby("image_path", sort=False):
        scanpaths = []
        for _, row in group.iterrows():
            locs = np.array(row["locations"], dtype=np.float32).reshape(-1, 2)
            scanpaths.append(locs)
        grouped[img_path] = scanpaths
    return grouped


# --- 224-grid lift, verbatim from scripts/evaluate_dg3_compat.py -----------
# (copied rather than imported: that module pulls in the DINO model stack,
# which the generator's environment does not carry. Keep in sync.)

def resize_log_predictions(log_preds: torch.Tensor, target_size: int) -> torch.Tensor:
    """exp -> bilinear interpolation -> renormalise -> log, i.e. probability
    mass pooled onto a target_size x target_size grid."""
    probs = log_preds.exp().unsqueeze(1)
    probs_resized = F.interpolate(
        probs, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    probs_resized = probs_resized.clamp_min(0)
    probs_resized = probs_resized / probs_resized.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(probs_resized.clamp_min(1e-12))


def coords_to_fixation_mask(xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    N = xy.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    col = ((x + 1.0) / 2.0 * (W - 1)).round().long().clamp(0, W - 1)
    row = ((y + 1.0) / 2.0 * (H - 1)).round().long().clamp(0, H - 1)
    mask = torch.zeros(N, H, W, dtype=torch.float32, device=xy.device)
    mask[torch.arange(N, device=xy.device), row, col] = 1.0
    return mask


def dg3_log_likelihood(log_density: torch.Tensor, fixation_mask: torch.Tensor) -> torch.Tensor:
    """DG3 LL formulation: (mean log-density at the fixation + log(H*W)) / log 2."""
    import math
    H, W = log_density.shape[-2], log_density.shape[-1]
    ll = (log_density * fixation_mask).sum(dim=(-1, -2)).mean()
    return (ll + math.log(H * W)) / math.log(2)


@torch.no_grad()
def evaluate_ceiling(
    model,
    val_data: dict,
    imagenet_root: str,
    device: str,
    batch_size: int = 16,
    heatmap_sigma: float = 2.0,
    eval_resolution: int = 0,
) -> dict:
    # Must match autoreg.utils.augmentations.build_augmentations_imagenet(...,
    # is_train=False): Resize(256) -> CenterCrop(224) -> tensor -> normalize.
    # The DINO model's own ScanpathDataset uses direct 224x224 resize, but that
    # would not faithfully replay the model that generated the parquet scanpaths.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    total_kl = 0.0
    total_ll = 0.0
    total_nss = 0.0
    total_auc = 0.0
    total_valid = 0
    n_images = 0
    # lifted to eval_resolution (DG3's 224 grid): DG3-formula LL, our NSS/AUC
    total_ll_lift = 0.0
    total_nss_lift = 0.0
    total_auc_lift = 0.0
    native_hw = None

    for img_path, scanpaths in tqdm(val_data.items(), desc="Evaluating"):
        full_path = Path(imagenet_root) / Path(img_path).relative_to("./data/ImageNet")
        if not full_path.exists():
            continue

        img = Image.open(full_path).convert("RGB")
        img_tensor = transform(img).to(device)  # (3, 224, 224)

        # Batch all selected scanpaths for this image.
        for start in range(0, len(scanpaths), batch_size):
            batch_sp = scanpaths[start:start + batch_size]
            B = len(batch_sp)

            # locations_preset: (B, 16, 2): initial (0,0) + 15 glimpses
            locs = np.stack(batch_sp)  # (B, 16, 2)
            locations_preset = torch.tensor(locs, dtype=torch.float32, device=device)

            img_batch = img_tensor.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 3, 224, 224)

            outputs = model(img_batch, training=False, locations_preset=locations_preset)

            # heatmap_logits: (B, G, H, W) where G=15
            # heatmap_logits[:, i] is produced at step i from glimpse at locations[:, i]
            # and predicts the next fixation locations[:, i+1]
            hm = outputs.heatmap_logits  # (B, 15, H, W)
            G = hm.shape[1]
            H, W = hm.shape[2], hm.shape[3]

            # log-softmax over spatial dims
            log_probs = F.log_softmax(hm.view(B, G, -1), dim=-1).view(B, G, H, W)

            # Targets: locations[:, 1:16] (the next fixation for each step)
            targets = locations_preset[:, 1:G + 1]  # (B, G, 2)

            for step_i in range(G):
                lp = log_probs[:, step_i]  # (B, H, W)
                tgt = targets[:, step_i]   # (B, 2)

                target_heatmap = make_gaussian_heatmaps_batch(
                    tgt.unsqueeze(1), H, heatmap_sigma
                ).squeeze(1).float()

                n_valid = B
                total_kl += kl_loss(lp, target_heatmap).item() * n_valid
                total_ll += ll_score(lp, tgt).item() * n_valid
                total_nss += nss_score(lp, tgt).item() * n_valid
                total_auc += auc_score(lp, tgt).item() * n_valid
                total_valid += n_valid

                if eval_resolution > 0:
                    native_hw = (H, W)
                    lp_r = resize_log_predictions(lp, eval_resolution)
                    mask = coords_to_fixation_mask(tgt, eval_resolution, eval_resolution)
                    total_ll_lift += dg3_log_likelihood(lp_r, mask).item() * n_valid
                    total_nss_lift += nss_score(lp_r, tgt).item() * n_valid
                    total_auc_lift += auc_score(lp_r, tgt).item() * n_valid

        n_images += 1

    out = {
        "kl": total_kl / total_valid if total_valid else float("nan"),
        "ll": total_ll / total_valid if total_valid else float("nan"),
        "nss": total_nss / total_valid if total_valid else float("nan"),
        "auc": total_auc / total_valid if total_valid else float("nan"),
        "steps": total_valid,
        "images": n_images,
    }
    if eval_resolution > 0 and total_valid:
        out.update({
            "eval_resolution": eval_resolution,
            "native_hw": native_hw,
            "ll_lift": total_ll_lift / total_valid,
            "nss_lift": total_nss_lift / total_valid,
            "auc_lift": total_auc_lift / total_valid,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute NSS/AUC/LL ceiling from generating model")
    parser.add_argument("--meta-json", default=META_JSON)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--parquet-path", default=PARQUET)
    parser.add_argument("--imagenet-root", default=IMAGENET_ROOT)
    parser.add_argument("--val-seed", type=int, default=VAL_SEED)
    parser.add_argument("--val-n", type=int, default=VAL_N)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-scanpaths-total", type=int, default=None,
                        help="keep only the first N validation scanpath rows; use 51200 to mirror 400 train.py val batches with batch size 128")
    parser.add_argument("--max-scanpaths-per-image", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-list", default=None,
                        help="protocol split file (class/file.JPEG per line); "
                             "replaces the seed-3141 canonical validation "
                             "selection. final evaluation only for test_10k.txt.")
    parser.add_argument("--eval-resolution", type=int, default=0,
                        help="additionally lift the generator's heatmaps to this "
                             "grid (224 = DeepGaze III's) and report DG3-formula "
                             "LL plus our NSS/AUC there, the same lift the "
                             "ladder's ll_dg3grid uses (evaluate_dg3_compat.py)")
    args = parser.parse_args()

    print("Loading generating model...", flush=True)
    model = load_generating_model(args.meta_json, args.checkpoint, args.device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params, "
          f"temperature={model.heatmap_temperature.item():.1f}, "
          f"num_glimpses={model.num_glimpses}", flush=True)

    print("Loading canonical val data from parquet...", flush=True)
    val_data = load_val_data(
        args.parquet_path,
        seed=args.val_seed,
        n_val=args.val_n,
        max_images=args.max_images,
        max_scanpaths_total=args.max_scanpaths_total,
        max_scanpaths_per_image=args.max_scanpaths_per_image,
        image_list=args.image_list,
    )
    print(f"  {len(val_data)} images with scanpaths", flush=True)
    total_sp = sum(len(v) for v in val_data.values())
    print(f"  {total_sp} total scanpaths", flush=True)

    print("Evaluating ceiling metrics...", flush=True)
    metrics = evaluate_ceiling(
        model, val_data, args.imagenet_root, args.device,
        batch_size=args.batch_size,
        heatmap_sigma=args.heatmap_sigma,
        eval_resolution=args.eval_resolution,
    )

    print("\n" + "=" * 60, flush=True)
    print("GENERATING MODEL SCORES (same metrics as fixation-prediction eval)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Images evaluated : {metrics['images']}", flush=True)
    print(f"  Valid steps      : {metrics['steps']}", flush=True)
    print(f"  KL               : {metrics['kl']:.4f}", flush=True)
    print(f"  LL (bits > unif) : {metrics['ll']:.4f}", flush=True)
    print(f"  NSS              : {metrics['nss']:.4f}", flush=True)
    print(f"  AUC              : {metrics['auc']:.4f}", flush=True)
    if "ll_lift" in metrics:
        r = metrics["eval_resolution"]
        print(f"  native heatmap   : {metrics['native_hw'][0]}x{metrics['native_hw'][1]}", flush=True)
        print(f"  LL@{r} (bits, DG3 grid) : {metrics['ll_lift']:.4f}", flush=True)
        print(f"  NSS@{r}          : {metrics['nss_lift']:.4f}", flush=True)
        print(f"  AUC@{r}          : {metrics['auc_lift']:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
