#!/usr/bin/env python3
"""Compute per-image duration-free MultiMatch and ScanMatch distributions.

Unlike the historical scorer, JSON ``size`` is interpreted as [width, height].
The output retains scores per image so plotting can use image bootstrap CIs and
fixed, shared histogram bins across every condition.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

SD_ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
sys.path.insert(0, str(SD_ROOT))

from src.gazetools.metrics.multimatch import multi_match_score
from src.gazetools.metrics.scanmatch import _scan_match_score

import GazeParser.ScanMatch as gsm

gsm.ScanMatch.CreateSubMatrix = gsm.ScanMatch.createSubMatrix
gsm.ScanMatch.GridMask = gsm.ScanMatch.gridMask


def generated_to_pixels(scanpaths, width: int, height: int):
    converted = []
    for sp in scanpaths:
        sp = np.asarray(sp)
        x = (sp[:, 0] + 1.0) * 0.5 * (width - 1)
        y = (sp[:, 1] + 1.0) * 0.5 * (height - 1)
        converted.append({"X": x.tolist(), "Y": y.tolist(), "T": [100.0] * len(sp)})
    return converted


def rescale_human(scanpaths, src_width: int, src_height: int,
                  dst_width: int, dst_height: int):
    """Copy human paths onto an aspect-preserving evaluation canvas."""
    sx = (dst_width - 1) / (src_width - 1)
    sy = (dst_height - 1) / (src_height - 1)
    converted = []
    for sp in scanpaths:
        item = dict(sp)
        item["X"] = (np.asarray(sp["X"], dtype=float) * sx).tolist()
        item["Y"] = (np.asarray(sp["Y"], dtype=float) * sy).tolist()
        converted.append(item)
    return converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    human = json.load(open(args.human))
    generated = np.load(args.generated, allow_pickle=True).item()
    keys = sorted(set(human) & set(generated))
    result = {}

    for index, key in enumerate(keys):
        width, height = map(int, human[key]["size"])
        # GazeParser 0.11.1 fails for resolutions that do not divide cleanly
        # into its fixed 16x12 grid (e.g. 500x333). Rescale each axis to the
        # next divisible size; this retains every image and changes the aspect
        # ratio by less than one grid cell.
        metric_width = math.ceil(width / 16) * 16
        metric_height = math.ceil(height / 12) * 12
        real = rescale_human(human[key]["scanpaths"], width, height,
                             metric_width, metric_height)
        model = generated_to_pixels(generated[key], metric_width, metric_height)

        model_mm = np.asarray([
            multi_match_score(r, g, size=(metric_width, metric_height))
            for r, g in itertools.product(real, model)
        ], dtype=np.float64)
        human_mm = np.asarray([
            multi_match_score(a, b, size=(metric_width, metric_height))
            for a, b in itertools.combinations(real, 2)
        ], dtype=np.float64)
        model_mm = model_mm[~np.isnan(model_mm).any(axis=1), :4]
        human_mm = human_mm[~np.isnan(human_mm).any(axis=1), :4]

        matcher = gsm.ScanMatch(
            Xres=metric_width, Yres=metric_height, Xbin=16, Ybin=12, TempBin=0
        )
        model_sm = np.asarray(_scan_match_score(matcher, real, model), dtype=np.float64)
        human_sm = np.asarray(_scan_match_score(matcher, real, None), dtype=np.float64)
        model_sm = model_sm[np.isfinite(model_sm)]
        human_sm = human_sm[np.isfinite(human_sm)]

        result[key] = {
            "model_mm": model_mm,
            "human_mm": human_mm,
            "model_sm": model_sm,
            "human_sm": human_sm,
            "size": (metric_width, metric_height),
            "n_model_paths": len(model),
            "n_human_paths": len(real),
        }
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(keys)} images", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out}: {len(result)} images")


if __name__ == "__main__":
    main()
