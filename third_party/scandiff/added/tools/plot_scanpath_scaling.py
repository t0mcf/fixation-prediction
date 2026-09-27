"""
Plot our model's scanpath-realism scaling: sequence-metric quality vs #synthetic
training images, from the CSV written by score_scanpaths.py (--csv).

Two views (the intuitive one + the table-comparable one):
  - raw MultiMatch / ScanMatch SIMILARITY of model-vs-human, with the human-vs-human
    ceiling as a dashed reference (does scanpath realism approach human with more data?)
  - KLD(human||model) of the score distributions (lower = better; comparable to ScanDiff's table)

Later, ScanDiff ladder rows (same CSV, label prefix 'scandiff') overlay as a second series.

Usage:
  python tools/plot_scanpath_scaling.py --csv data/eval/scaling.csv --out data/eval/viz/scanpath_scaling.png
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OURS, SCAND, GREY = "#2563EB", "#E08A2B", "#9AA0A6"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 13, "axes.labelsize": 13.5, "axes.labelweight": "medium",
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "axes.edgecolor": "#444",
    "axes.grid": True, "grid.color": "#E6E6E6", "font.family": "DejaVu Sans"})


def kfmt(x, _):
    if x >= 1_000_000: return f"{x/1e6:g}M"
    if x >= 1_000:     return f"{x/1e3:g}k"
    return f"{x:g}"


def load(csv_path):
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("n_images") not in (None, "", "None")]
    for r in rows:
        for k in ("n_images", "mm_kld_nodur", "mm_sim_model", "mm_sim_human",
                  "sm_kld", "sm_model", "sm_human"):
            r[k] = float(r[k])
    rows.sort(key=lambda r: r["n_images"])
    return rows


def series(rows, prefix):
    sub = [r for r in rows if r["label"].startswith(prefix)]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/eval/viz/scanpath_scaling.png")
    args = ap.parse_args()
    rows = load(args.csv)
    ours = series(rows, "ours")
    scand = series(rows, "scandiff")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), constrained_layout=True)

    # Top row: raw similarity to human (intuitive, with human-human ceiling).
    # Bottom row: KLD of the score distributions (lower=better; matches ScanDiff's table).
    panels = [
        (axes[0, 0], "sim", "MultiMatch", "mm_sim_model", "mm_sim_human",
         "MultiMatch — similarity to human"),
        (axes[0, 1], "sim", "ScanMatch", "sm_model", "sm_human",
         "ScanMatch — similarity to human"),
        (axes[1, 0], "kld", "MultiMatch", "mm_kld_nodur", None,
         "MultiMatch — KLD(human‖model) ↓"),
        (axes[1, 1], "kld", "ScanMatch", "sm_kld", None,
         "ScanMatch — KLD(human‖model) ↓"),
    ]
    for ax, kind, metric, yk, ceilk, title in panels:
        if ours:
            ox = [r["n_images"] for r in ours]; oy = [r[yk] for r in ours]
            ax.plot(ox, oy, "o-", color=OURS, lw=2.8, ms=8.5, label="Ours", zorder=4)
            ax.annotate(f"{oy[-1]:.3f}", (ox[-1], oy[-1]), textcoords="offset points",
                        xytext=(6, 4), color=OURS, fontweight="bold", fontsize=11)
            if kind == "sim" and ceilk:
                ceil = sum(r[ceilk] for r in ours) / len(ours)
                ax.axhline(ceil, color=GREY, ls=(0, (5, 2)), lw=2, zorder=2)
                ax.annotate(f"human–human ceiling ({ceil:.3f})", (ox[0], ceil),
                            textcoords="offset points", xytext=(0, 6), color="#666", fontsize=10.5)
        if scand:
            sx = [r["n_images"] for r in scand]; sy = [r[yk] for r in scand]
            ax.plot(sx, sy, "o-", color=SCAND, lw=2.6, ms=8, mfc="white", mec=SCAND,
                    mew=2, label="ScanDiff", zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        if ours:
            ax.set_xticks([r["n_images"] for r in ours])
            ax.set_xticklabels([kfmt(r["n_images"], None) for r in ours], fontsize=11)
        ax.grid(axis="x", alpha=0.0); ax.set_axisbelow(True); ax.tick_params(length=4, color="#888")
        ax.set_xlabel("synthetic training images")
        ax.set_ylabel(f"{metric} similarity to human" if kind == "sim"
                      else f"{metric} KLD (lower better)")
        ax.set_title(title, fontsize=12.5)
        if ours or scand:
            ax.legend(loc="lower right" if kind == "sim" else "upper left", frameon=False)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    print(f"saved {out}")
    print(f"ours points: {[(int(r['n_images']), round(r['mm_sim_model'],3)) for r in ours]}")


if __name__ == "__main__":
    main()
