"""
Zero-shot evaluation of DG3 imagenet-pretrained checkpoints on MIT1003.

Models were trained only on synthetic ImageNet scanpaths.
Evaluates NSS / AUC / LL on the full MIT1003 human fixation dataset.

Usage:
    python -m scanpath.eval_zeroshot_mit1003 \
        --checkpoints runs/imagenet_paths/dg3_100k_5sp/scanpath/best.pth \
        --names "DG3-100k" \
        --dataset-dir /mnt/vast-nhr/projects/nim00018/datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pysaliency

from deepgaze_pytorch.metrics import log_likelihood as _log_likelihood, nss as _nss, auc as _auc

from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
from deepgaze_pytorch.features.densenet import RGBDenseNet201

from scanpath.deepgaze_engine import (
    make_scanpath_loader,
    convert_fixation_trains,
    convert_stimuli,
    build_saliency_network,
    build_scanpath_network,
    build_fixation_selection_network,
)
from scanpath.train_deepgaze import _ZeroCenterbias


def load_model(checkpoint_path: str, device: str) -> DeepGazeIII:
    """Load a DG3 scanpath model from an imagenet-style checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    densenet_layers = [
        "1.features.denseblock4.denselayer32.norm1",
        "1.features.denseblock4.denselayer32.conv1",
        "1.features.denseblock4.denselayer31.conv2",
    ]
    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), densenet_layers),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=1,
        readout_factor=2.25,
        saliency_map_factor=1,
        included_fixations=[-1, -2, -3, -4],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_on_loader(model, loader, device: str) -> dict:
    nss_scores, auc_scores, ll_scores, weights_all = [], [], [], []

    for batch in loader:
        image         = batch.pop("image").to(device)
        centerbias    = batch.pop("centerbias").to(device)
        fixation_mask = batch.pop("fixation_mask").to(device)
        x_hist        = batch.pop("x_hist", torch.tensor([])).to(device)
        y_hist        = batch.pop("y_hist", torch.tensor([])).to(device)
        weights       = batch.pop("weight").to(device)
        durations     = batch.pop("durations", torch.tensor([])).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_density = model(
                image, centerbias,
                x_hist=x_hist, y_hist=y_hist, durations=durations,
            ).float()

        w = weights.detach().cpu().numpy().sum()
        nss_scores.append(_nss(log_density, fixation_mask, weights=weights).detach().cpu().item())
        auc_scores.append(_auc(log_density, fixation_mask, weights=weights).detach().cpu().item())
        ll_scores.append(_log_likelihood(log_density, fixation_mask, weights=weights).detach().cpu().item())
        weights_all.append(w)

    w = np.array(weights_all)
    return {
        "NSS": float(np.average(nss_scores, weights=w)),
        "AUC": float(np.average(auc_scores, weights=w)),
        "LL":  float(np.average(ll_scores,  weights=w)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names",        nargs="+", default=None)
    parser.add_argument("--dataset-dir",  default="/mnt/vast-nhr/projects/nim00018/datasets")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--num-workers",  type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names  = args.names or [Path(c).parent.parent.name for c in args.checkpoints]
    assert len(names) == len(args.checkpoints)

    # Load MIT1003
    print("Loading MIT1003 …", flush=True)
    dataset_dir = Path(args.dataset_dir)
    stimuli_orig, scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=dataset_dir, replace_initial_invalid_fixations=True,
    )

    # Convert BOTH coords and images to the two-size standard (768x1024 / 1024x768)
    # so the fixation mask grid matches the image dimensions. This mirrors exactly
    # what run_mit1003 does — convert_fixation_trains MUST be paired with
    # convert_stimuli, not preprocess_to_resized_filestimuli (which forces 224x224).
    twosize_dir = Path(args.dataset_dir) / "MIT1003_twosize_eval"
    scanpaths = convert_fixation_trains(stimuli_orig, scanpaths_orig)
    stimuli   = convert_stimuli(stimuli_orig, twosize_dir)
    fixations = scanpaths[scanpaths.lengths > 0]

    centerbias = _ZeroCenterbias()

    loader = make_scanpath_loader(
        stimuli, fixations, centerbias,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_data=False,
    )
    print(f"MIT1003 loader ready: {len(fixations)} fixations", flush=True)

    print(f"\n{'Model':<20} {'NSS':>8} {'AUC':>8} {'LL':>8}")
    print("-" * 50)

    for ckpt_path, name in zip(args.checkpoints, names):
        print(f"loading {ckpt_path} …", flush=True)
        model   = load_model(ckpt_path, device)
        metrics = evaluate_on_loader(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(
            f"{name:<20} {metrics['NSS']:>8.3f} {metrics['AUC']:>8.3f} {metrics['LL']:>8.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
