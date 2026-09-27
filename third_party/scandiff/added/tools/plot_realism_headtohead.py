"""
Head-to-head scanpath-realism scaling figure: our model (T=1.0 and T=0.6)
vs ScanDiff (length-matched protocol), on MIT1003.

Top row: raw mean similarity vs human (higher = better) with the human-human
ceiling; bottom row: KLD(human || model) of the score distributions
(lower = better). Left: MultiMatch (duration-free, 4-dim);
right: ScanMatch (no duration).

Data:
  ours T=1.0 : data/eval/scaling.csv        (full 1003-image eval set)
  ours T=0.6 : data/eval/scaling_T06.csv    (full 1003-image eval set)
  scandiff   : data/eval/scandiff_ladder.csv, '-lenmatch' rows (100-image set,
               1k-200k, all converged final checkpoints)

Usage:
  python tools/plot_realism_headtohead.py --out data/eval/viz/realism_headtohead.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
BLUE, BLUE_LT, ORANGE, GREY = "#2a78d6", "#7fabe3", "#eb6834", "#9AA0A6"
INK = "#0b0b0b"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 12.5, "axes.labelsize": 13, "axes.labelweight": "medium",
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "axes.edgecolor": "#444444",
    "axes.grid": True, "grid.color": "#ECECEC", "font.family": "DejaVu Sans"})


def kfmt(x, _):
    return f"{x/1e6:g}M" if x >= 1e6 else (f"{x/1e3:g}k" if x >= 1e3 else f"{x:g}")


def load(path, want=None, exclude=None):
    rows = []
    for r in csv.DictReader(open(path)):
        if not r.get("n_images"):
            continue
        lab = r["label"]
        if want and want not in lab:
            continue
        if exclude and exclude in lab:
            continue
        rows.append({k: (float(r[k]) if k != "label" else r[k])
                     for k in ("label", "n_images", "mm_kld_nodur", "mm_sim_model",
                               "mm_sim_human", "sm_kld", "sm_model", "sm_human")})
    rows.sort(key=lambda r: r["n_images"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/eval/viz/realism_headtohead.png"))
    args = ap.parse_args()

    ours_t1 = load(ROOT / "data/eval/scaling.csv")
    ours_t06 = load(ROOT / "data/eval/scaling_T06.csv")
    sd = load(ROOT / "data/eval/scandiff_ladder.csv", want="lenmatch")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    panels = [
        ("mm_sim_model", "mm_sim_human", "MultiMatch similarity (4-dim, no duration)", True),
        ("sm_model", "sm_human", "ScanMatch similarity (no duration)", True),
        ("mm_kld_nodur", None, "MultiMatch KLD(human ∥ model)  ↓", False),
        ("sm_kld", None, "ScanMatch KLD(human ∥ model)  ↓", False),
    ]
    series_specs = [
        (ours_t1, BLUE, "-", "ours (T=1.0)"),
        (ours_t06, BLUE_LT, "--", "ours (T=0.6)"),
        (sd, ORANGE, "-", "ScanDiff (len-matched)"),
    ]
    handles = None
    for ax, (key, ceil_key, title, is_sim) in zip(axes.flat, panels):
        for rows, color, ls, lab in series_specs:
            x = [r["n_images"] for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, ls, color=color, lw=2, marker="o", ms=6,
                    mfc=color, mec="white", mew=1.2, label=lab, zorder=3)
        if ceil_key:
            ceil = ours_t1[0][ceil_key]
            ax.axhline(ceil, color=GREY, lw=1.6, ls=(0, (4, 3)), zorder=2,
                       label="human–human ceiling" if handles is None else None)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        # headroom above the highest series so the legend never sits on data
        ys_all = [v for rows, *_ in series_specs for r in rows for v in [r[key]]]
        if ceil_key:
            ys_all.append(ours_t1[0][ceil_key])
        lo, hi = min(ys_all), max(ys_all)
        pad = (hi - lo) * 0.22 if hi > lo else 0.02
        ax.set_ylim(lo - pad * 0.3, hi + pad)
        ax.set_title(title, fontsize=12.5, color=INK, pad=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    for ax in axes[1]:
        ax.set_xlabel("synthetic training images")

    fig.suptitle("Scanpath realism vs training scale — ours vs ScanDiff (MIT1003)",
                 fontsize=14.5, fontweight="medium", color=INK, y=0.995)
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 0.965), fontsize=11)
    fig.text(0.01, 0.005,
             "Ours: full 1003-image eval, sampler given human lengths. ScanDiff: 100-image subset, "
             "generations truncated to the same human lengths (equal length oracle). "
             "All ScanDiff points are converged final checkpoints (1k–200k).",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.025, 1, 0.90))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
