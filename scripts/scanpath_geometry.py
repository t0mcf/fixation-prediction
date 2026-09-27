"""Scanpath geometry of the three sources, recomputed from the data.

Every source is converted to the same convention before anything is measured:
normalised image coordinates in [-1,1], centre of the image at (0,0).

Sources
  human     MIT1003, all 1,003 images (descriptive statistics of the dataset),
            x/width and y/height
  av        the AV pretraining corpus of the source comparison
            (av_fixed5_full_v2u_100k), complete, already in [-1,1]
  scandiff  the ScanDiff-generated pretraining corpus of the source comparison
            (scandiffgen_official_v2_avcrop_100k_unfiltered), complete,
            already in [-1,1]

Both synthetic sources are the EXACT training corpora (all ~500k scanpaths on
the same 100k images), enumerated fully -- no sampling.

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
import json
from pathlib import Path

import numpy as np

SCANDIFF = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
# All 1,003 images: these are descriptive statistics of a public dataset,
# not a model result, so they inform no decision and may include the test
# split (user decision 2026-09-15; the 2026-09-14 run used train+validation
# only, 12,676 paths — numbers differ in the third decimal).
MIT_JSONS = [SCANDIFF / "paper_reproduction/data" / f"mit1003_fixations_{s}.json"
             for s in ("train", "validation", "test")]
SD_PARQUET = SCANDIFF / "data/scandiffgen_official_v2_avcrop_100k_unfiltered/merged.parquet"
AV_PARQUET = SCANDIFF / "data/av_fixed5_full_v2u_100k/merged.parquet"
MAX_INDEX = 15


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


def corpus_scanpaths(parquet: Path):
    """One of the two matched pretraining corpora, enumerated in full.

    Both store variable-length scanpaths as flat [-1,1] coordinate arrays on
    the square 224 crop; no padding sentinel (lengths are exact).
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet)
    out = []
    for group in range(handle.num_row_groups):
        table = handle.read_row_group(group, columns=["locations"])
        for row in table.column("locations").to_pylist():
            xy = np.asarray(row, dtype=np.float32).reshape(-1, 2)
            if len(xy) >= 2:
                out.append((xy, 1.0))
    return out


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
    """scanpaths: [(xy in [-1,1], aspect)]. Reports raw and isotropic figures,
    plus diagonal-normalised ones (distances as a fraction of the image
    diagonal, the same convention MultiMatch uses; handles mixed aspect
    ratios per record)."""
    points = np.concatenate([sp for sp, _ in scanpaths])
    steps = [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp, _ in scanpaths]
    iso = [isotropic(sp, a) for sp, a in scanpaths]
    iso_points = np.concatenate(iso)
    iso_steps = np.concatenate(
        [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp in iso])
    # iso coords are in half-image-heights; the diagonal in those units is
    # 2*sqrt(1 + aspect^2), so dividing per record converts to diag fraction.
    diag = [sp / (2.0 * np.sqrt(1.0 + a * a))
            for sp, a in zip(iso, (a for _, a in scanpaths))]
    diag_points = np.concatenate(diag)
    diag_steps = np.concatenate(
        [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp in diag])
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
        "x_sd_diag": float(diag_points[:, 0].std()),
        "mean_r_diag": float(np.linalg.norm(diag_points, axis=1).mean()),
        "sacc_diag": float(diag_steps.mean()),
        "sacc_by_index": by_index,
    }


def diag_arrays(scanpaths):
    """Per-source arrays in image-diagonal fractions: fixation xy, distance
    from centre, saccade amplitudes, and within-scanpath dispersion. Same
    transform chain as stats(). Dispersion is the RMS fixation distance from
    the scanpath's own centroid."""
    iso = [isotropic(sp, a) for sp, a in scanpaths]
    diag = [sp / (2.0 * np.sqrt(1.0 + a * a))
            for sp, a in zip(iso, (a for _, a in scanpaths))]
    points = np.concatenate(diag)
    steps = np.concatenate(
        [np.linalg.norm(np.diff(sp, axis=0), axis=1) for sp in diag])
    dispersion = np.asarray([
        np.sqrt(np.mean(np.sum((sp - sp.mean(axis=0)) ** 2, axis=1)))
        for sp in diag
    ])
    return points, np.linalg.norm(points, axis=1), steps, dispersion


def diag_scanpaths(scanpaths):
    """[(xy, aspect)] -> list of scanpaths in isotropic image-diagonal units."""
    return [isotropic(sp, a) / (2.0 * np.sqrt(1.0 + a * a)) for sp, a in scanpaths]


def sequence_arrays(scanpaths):
    """Sequence statistics of the kind DeepGaze III's Fig. 3 uses
    (Kümmerer, Bethge & Wallis 2022):
      direction   angle of every saccade vector, degrees in [0, 360), 0 = to the
                  right, 90 = up (image y axis points down, so dy is negated);
                  computed on isotropic coordinates so a 4:3 photograph does
                  not squash horizontal saccades.
      angle       SIGNED angle between consecutive saccade vectors, degrees
                  in (-180, 180]: 0 = same direction, +-180 = return saccade,
                  positive = counter-clockwise turn (y up) — the DeepGaze III
                  Fig. 3e/f convention (changed 2026-09-17; was unsigned).
      amp_pairs   {lag: (a_n, a_{n+lag})} amplitude pairs within scanpaths,
                  for the amplitude autocorrelation (Pearson r per lag).
    """
    directions, angles = [], []
    pairs = {1: ([], []), 2: ([], []), 3: ([], []), 4: ([], [])}
    for sp in diag_scanpaths(scanpaths):
        v = np.diff(sp, axis=0)                      # saccade vectors
        if len(v) == 0:
            continue
        amp = np.linalg.norm(v, axis=1)
        keep = amp > 0
        directions.append(np.degrees(np.arctan2(-v[keep, 1], v[keep, 0])) % 360.0)
        if len(v) >= 2:
            a, b = v[:-1], v[1:]
            na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            ok = (na > 0) & (nb > 0)
            dot = np.sum(a[ok] * b[ok], axis=1)
            # image y points down; negate so a positive angle is a
            # counter-clockwise turn with y up (same convention as 'direction')
            cross = -(a[ok, 0] * b[ok, 1] - a[ok, 1] * b[ok, 0])
            angles.append(np.degrees(np.arctan2(cross, dot)))
        for lag, (xs, ys) in pairs.items():
            if len(amp) > lag:
                xs.append(amp[:-lag]); ys.append(amp[lag:])
    out = {"direction": np.concatenate(directions) if directions else np.empty(0),
           "angle": np.concatenate(angles) if angles else np.empty(0)}
    for lag, (xs, ys) in pairs.items():
        x = np.concatenate(xs) if xs else np.empty(0); y = np.concatenate(ys) if ys else np.empty(0)
        out[f"autocorr_lag{lag}"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        out[f"n_pairs_lag{lag}"] = int(len(x))
    d = out["direction"]
    # share of saccades within +-30 deg of horizontal / vertical
    out["horizontal_frac"] = float(np.mean((np.minimum(d, 360 - d) <= 30) | (np.abs(d - 180) <= 30))) if len(d) else float("nan")
    out["vertical_frac"] = float(np.mean((np.abs(d - 90) <= 30) | (np.abs(d - 270) <= 30))) if len(d) else float("nan")
    ang = np.abs(out["angle"])
    # shares on the UNSIGNED angle (the signed value is only for the histogram):
    # a clockwise reversal (-170) is a reversal too.
    aang = np.abs(ang)
    out["return_frac"] = float(np.mean(aang >= 150)) if len(ang) else float("nan")   # near-antiparallel
    out["forward_frac"] = float(np.mean(aang <= 30)) if len(ang) else float("nan")
    return out


HIST_SPECS = {
    # stat -> (bin edges). All in image-diagonal fractions except length.
    "sacc": np.linspace(0.0, 0.75, 76),
    "r": np.linspace(0.0, 0.60, 61),
    "dispersion": np.linspace(0.0, 0.50, 61),
    "direction": np.linspace(0.0, 360.0, 73),    # 5-degree bins, density per degree
    "angle": np.linspace(-180.0, 180.0, 73),     # 5-degree bins, signed
}


def write_hists(sources, path: Path):
    """Binned distributions for the density figure: saccade amplitude,
    distance from centre and x position (diagonal fractions, *_nofirst
    variants), plus the scanpath-length distribution (full paths, as
    trained). Densities integrate to 1 (length: shares sum to 1)."""
    rows = []
    for base in ("human", "av", "scandiff"):
        points, r, steps, dispersion = diag_arrays(sources[base + "_nofirst"])
        seq = sequence_arrays(sources[base + "_nofirst"])
        for stat, values in (("sacc", steps), ("r", r),
                             ("dispersion", dispersion),
                             ("direction", seq["direction"]),
                             ("angle", seq["angle"])):
            edges = HIST_SPECS[stat]
            dens, _ = np.histogram(values, bins=edges, density=True)
            for left, right, d in zip(edges[:-1], edges[1:], dens):
                rows.append({"source": base, "stat": stat,
                             "bin_left": round(float(left), 4),
                             "bin_right": round(float(right), 4),
                             "density": float(d)})
        lengths = np.array([len(sp) for sp, _ in sources[base]])
        for n in range(1, 17):
            rows.append({"source": base, "stat": "length",
                         "bin_left": n, "bin_right": n + 1,
                         "density": float((lengths == n).mean())})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/report_tables/scanpath_geometry.csv")
    parser.add_argument("--hist-out",
                        default="docs/report_tables/scanpath_geometry_hists.csv")
    args = parser.parse_args()

    sources = {
        "human": human_scanpaths(),
        "av": corpus_scanpaths(AV_PARQUET),
        "scandiff": corpus_scanpaths(SD_PARQUET),
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

    rows = {}
    for name, paths in sources.items():
        rows[name] = stats(paths)
        seq = sequence_arrays(paths)
        rows[name].update({k: seq[k] for k in seq if not isinstance(seq[k], np.ndarray)})
        s = rows[name]
        print(f"{name:12s} n={s['n_scanpaths']:6d} len={s['mean_length']:5.2f} | "
              f"raw  x_sd={s['x_sd']:.4f} r={s['mean_r']:.4f} sacc={s['sacc_pooled']:.4f} | "
              f"iso  x_sd={s['x_sd_iso']:.4f} r={s['mean_r_iso']:.4f} sacc={s['sacc_iso']:.4f}")

    print("\nsequence statistics (autocorr lag1 | horizontal | return-saccade share):")
    for name, s2 in rows.items():
        print(f"  {name:16s} r1={s2['autocorr_lag1']:+.3f} (n={s2['n_pairs_lag1']}) "
              f"horiz={s2['horizontal_frac']:.2f} vert={s2['vertical_frac']:.2f} "
              f"return={s2['return_frac']:.2f} forward={s2['forward_frac']:.2f}")
    print("\nmean saccade length by ordinal position (1..8):")
    for name, s in rows.items():
        print(f"  {name:12s} " + " ".join(f"{v:.3f}" for v in s["sacc_by_index"][:8]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "n_scanpaths", "n_fixations", "mean_length",
              "x_sd", "y_sd", "mean_r", "sacc_pooled", "sacc_per_path",
              "x_sd_iso", "mean_r_iso", "sacc_iso",
              "x_sd_diag", "mean_r_diag", "sacc_diag",
              "autocorr_lag1", "autocorr_lag2", "autocorr_lag3",
              "n_pairs_lag1", "horizontal_frac", "vertical_frac",
              "return_frac", "forward_frac"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, s in rows.items():
            writer.writerow({"source": name, **{k: s[k] for k in fields[1:]}})
    print(f"\nwrote {out}")

    write_hists(sources, Path(args.hist_out))


if __name__ == "__main__":
    main()
