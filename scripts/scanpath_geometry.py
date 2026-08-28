"""Scanpath geometry of the three sources, recomputed from the data.

Every source is converted to the same convention before anything is measured:
normalised image coordinates in [-1,1], centre of the image at (0,0).

Sources
  human     MIT1003, all three of the authors' splits, x/width and y/height
  av        the active-vision corpus, already stored in [-1,1]
  scandiff  the protocol-v2 generations from the official checkpoint, stored in
            [0,1] with -3 padding, mapped to [-1,1] here

Statistics, per source
  x_sd, y_sd          spread of fixations across the image
  mean_r              mean distance from the image centre -- the central bias
  sacc_pooled         mean saccade length over every consecutive pair
  sacc_per_path       mean saccade length per scanpath, then averaged over
                      scanpaths, so a long scanpath does not outweigh a short one
  sacc_by_index       mean saccade length at ordinal position 1, 2, 3, ...

sacc_per_path and sacc_by_index control for the very different scanpath
lengths across sources (active vision 16, ScanDiff ~8.5 on average).

  python scripts/scanpath_geometry.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np
import torch

SCANDIFF = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
MIT_JSONS = sorted((SCANDIFF / "paper_reproduction/data").glob("mit1003_fixations_*.json"))
SD_GLOB = ("runs_ours/paper_official_avcrop_v2_100k_oursynth_v2_100k_shard*/"
           "generations_epoch_0/generations_MIT1003Dataset_validation.pkl")
AV_PARQUET = Path("/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/"
                  "2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet")
PAD = -3.0
MAX_INDEX = 15


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def human_scanpaths():
    """MIT1003 fixations, normalised by each record's own width and height.

    Returns (xy, aspect) with aspect = width/height. MIT1003 images are mostly
    1024x768, so x and y are divided by different numbers and one normalised
    unit is not the same physical distance in the two axes. The synthetic
    sources live on a square crop where it is. Carrying the aspect lets the
    isotropic statistics undo that.
    """
    out = []
    for path in MIT_JSONS:
        for record in json.loads(path.read_text()):
            xy = np.column_stack([
                np.asarray(record["X"], dtype=float) / record["width"],
                np.asarray(record["Y"], dtype=float) / record["height"],
            ])
            if len(xy) >= 2:
                out.append((xy * 2.0 - 1.0, record["width"] / record["height"]))
    return out


def av_scanpaths(row_group_step, per_group, drop_center):
    """Active-vision scanpaths, already in [-1,1] on a square crop.

    Takes at most per_group scanpaths from each visited row group. The earlier
    version appended whole row groups and stopped once it had enough, which
    meant everything came from the first two groups -- and the parquet is
    sorted by ImageNet class, so that was two synsets rather than a sample of
    the corpus.

    drop_center removes the first fixation, which the active-vision protocol
    pins to the exact image centre. It is a protocol artefact, not a fixation
    the policy chose, and it pulls the central-bias statistic toward zero.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(AV_PARQUET)
    out = []
    for group in range(0, handle.num_row_groups, row_group_step):
        table = handle.read_row_group(group, columns=["locations"])
        for row in table.column("locations").to_pylist()[:per_group]:
            xy = np.asarray(row, dtype=np.float32).reshape(-1, 2)
            if drop_center:
                xy = xy[1:]
            if len(xy) >= 2:
                out.append((xy, 1.0))
    return out


def scandiff_scanpaths(shards, keep_oob=False):
    """Protocol-v2 generations, mapped [0,1] -> [-1,1].

    keep_oob controls whether sequences leaving the image are kept. This has
    to describe the data a model would actually be trained on, not the raw
    sampler output — so it must match the training corpus: --keep-oob for the
    unfiltered v2 corpus (out-of-bounds sequences kept in),
    default drop for the earlier filtered corpus.
    """
    kept, rejected = [], 0
    for path in sorted(SCANDIFF.glob(SD_GLOB))[:shards]:
        with path.open("rb") as handle:
            generations = _CpuUnpickler(handle).load()
        for record in generations.values():
            for sp in np.asarray(record["scanpaths"]):
                keep = ~np.any(sp[:, :2] <= PAD + 1e-3, axis=1)
                xy = sp[keep, :2]
                if len(xy) < 2:
                    continue
                if not keep_oob and ((xy < 0) | (xy > 1)).any():
                    rejected += 1
                    continue
                kept.append((xy * 2.0 - 1.0, 1.0))
    return kept, rejected


def isotropic(xy, aspect):
    """Rescale x so that one unit means the same distance in both axes.

    Distances then read in units of image height. For the square synthetic
    crops aspect is 1 and nothing changes; for a 4:3 photograph it undoes the
    horizontal compression that plain [-1,1] normalisation introduces --
    which matters because human saccades are predominantly horizontal.
    """
    if aspect == 1.0:
        return xy
    scaled = xy.copy()
    scaled[:, 0] *= aspect
    return scaled


def stats(scanpaths):
    """scanpaths: [(xy in [-1,1], aspect)]. Reports raw and isotropic figures."""
    points = np.concatenate([sp for sp, _ in scanpaths])
    steps = [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp, _ in scanpaths]
    iso = [isotropic(sp, a) for sp, a in scanpaths]
    iso_points = np.concatenate(iso)
    iso_steps = np.concatenate(
        [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp in iso])
    pooled = np.concatenate(steps)
    by_index = []
    for i in range(MAX_INDEX):
        at_i = [s[i] for s in steps if len(s) > i]
        by_index.append(float(np.mean(at_i)) if at_i else float("nan"))
    return {
        "n_scanpaths": len(scanpaths),
        "n_fixations": len(points),
        "mean_length": float(np.mean([len(sp) for sp, _ in scanpaths])),
        "x_sd": float(points[:, 0].std()),
        "y_sd": float(points[:, 1].std()),
        "mean_r": float(np.linalg.norm(points, axis=1).mean()),
        "sacc_pooled": float(pooled.mean()),
        "sacc_per_path": float(np.mean([s.mean() for s in steps])),
        "x_sd_iso": float(iso_points[:, 0].std()),
        "mean_r_iso": float(np.linalg.norm(iso_points, axis=1).mean()),
        "sacc_iso": float(iso_steps.mean()),
        "sacc_by_index": by_index,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/report_tables/scanpath_geometry.csv")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--av-row-group-step", type=int, default=2)
    parser.add_argument("--av-per-group", type=int, default=300)
    parser.add_argument("--keep-oob", action="store_true",
                        help="keep out-of-bounds ScanDiff sequences (matches "
                             "the unfiltered v2 training corpus)")
    args = parser.parse_args()

    sd, rejected = scandiff_scanpaths(args.shards, keep_oob=args.keep_oob)
    sources = {
        "human": human_scanpaths(),
        "av": av_scanpaths(args.av_row_group_step, args.av_per_group, False),
        "scandiff": sd,
    }
    # MIT1003 recordings begin on a central fixation cross (83% of first
    # fixations sit within 0.1 of the centre) and the active-vision protocol
    # pins its first fixation to exactly (0,0). ScanDiff has no forced start at
    # all. Leaving the first fixation in therefore drags the central-bias and
    # first-saccade figures down for two sources out of three. The _nofirst
    # rows drop it from every source alike, which is the comparable variant.
    for name in list(sources):
        sources[name + "_nofirst"] = [(sp[1:], a) for sp, a in sources[name]
                                      if len(sp) >= 3]
    print(f"scandiff: dropped {rejected} out-of-bounds sequences\n")

    rows = {}
    for name, paths in sources.items():
        rows[name] = stats(paths)
        s = rows[name]
        print(f"{name:12s} n={s['n_scanpaths']:6d} len={s['mean_length']:5.2f} | "
              f"raw  x_sd={s['x_sd']:.4f} r={s['mean_r']:.4f} sacc={s['sacc_pooled']:.4f} | "
              f"iso  x_sd={s['x_sd_iso']:.4f} r={s['mean_r_iso']:.4f} sacc={s['sacc_iso']:.4f}")

    print("\nmean saccade length by ordinal position (1..8):")
    for name, s in rows.items():
        print(f"  {name:12s} " + " ".join(f"{v:.3f}" for v in s["sacc_by_index"][:8]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "n_scanpaths", "n_fixations", "mean_length",
              "x_sd", "y_sd", "mean_r", "sacc_pooled", "sacc_per_path",
              "x_sd_iso", "mean_r_iso", "sacc_iso"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, s in rows.items():
            writer.writerow({"source": name, **{k: s[k] for k in fields[1:]}})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
