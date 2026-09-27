#!/usr/bin/env python3
"""Aggregate corrected per-image sequence scores and create publication plots."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import entropy

BINS = np.linspace(0.0, 1.0, 101)
COLORS = {"ours": "#2a78d6", "DeepGaze III": "#eb6834", "ScanDiff": "#2e9e6b"}
MARKERS = {"ours": "o", "DeepGaze III": "s", "ScanDiff": "D"}


def hist_kld(human, model):
    hp = np.histogram(human, bins=BINS)[0].astype(float)
    mp = np.histogram(model, bins=BINS)[0].astype(float)
    hp /= hp.sum()
    mp /= mp.sum()
    return float(entropy(hp, mp + 1e-10))


def aggregate(data, keys):
    hmm = np.concatenate([data[k]["human_mm"] for k in keys])
    mmm = np.concatenate([data[k]["model_mm"] for k in keys])
    hsm = np.concatenate([data[k]["human_sm"] for k in keys])
    msm = np.concatenate([data[k]["model_sm"] for k in keys])
    return {
        "mm_raw": float(mmm.mean()),
        "mm_human": float(hmm.mean()),
        "mm_kld": float(np.mean([hist_kld(hmm[:, d], mmm[:, d]) for d in range(4)])),
        "sm_raw": float(msm.mean()),
        "sm_human": float(hsm.mean()),
        "sm_kld": hist_kld(hsm, msm),
    }


def summarize(path, bootstraps, seed):
    with open(path, "rb") as f:
        data = pickle.load(f)
    keys = sorted(data)
    point = aggregate(data, keys)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("mm_raw", "mm_kld", "sm_raw", "sm_kld")}
    for _ in range(bootstraps):
        sampled = rng.choice(keys, size=len(keys), replace=True).tolist()
        values = aggregate(data, sampled)
        for key in draws:
            draws[key].append(values[key])
    for key, values in draws.items():
        point[key + "_lo"], point[key + "_hi"] = np.quantile(values, [0.025, 0.975])
    point["n_images"] = len(keys)
    return point


def kfmt(x, _):
    return f"{x/1e6:g}M" if x >= 1e6 else f"{x/1e3:g}k"


def plot(rows, out, title, normalized=False):
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 220, "figure.facecolor": "white",
        "font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ECECEC", "font.family": "DejaVu Sans",
    })
    panels = [
        ("mm_raw", "MultiMatch similarity (4D, no duration)", True),
        ("sm_raw", "ScanMatch similarity (no duration)", True),
        ("mm_kld", r"MultiMatch $D_{KL}(P_{HH}\,\Vert\,P_{HM})$", False),
        ("sm_kld", r"ScanMatch $D_{KL}(P_{HH}\,\Vert\,P_{HM})$", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for ax, (metric, label, higher) in zip(axes.flat, panels):
        for model in ("ours", "DeepGaze III", "ScanDiff"):
            subset = sorted([r for r in rows if r["model"] == model], key=lambda r: r["scale"])
            if not subset:
                continue
            x = np.array([r["scale"] for r in subset], float)
            y = np.array([r[metric] for r in subset], float)
            lo = np.array([r[metric + "_lo"] for r in subset], float)
            hi = np.array([r[metric + "_hi"] for r in subset], float)
            if normalized:
                base = y[0]
                y, lo, hi = y - base, lo - base, hi - base
            ax.errorbar(x, y, yerr=[y-lo, hi-y], color=COLORS[model], marker=MARKERS[model],
                        lw=2, ms=6, capsize=2.5, label=model, zorder=3)
        if not normalized and higher:
            ref = float(np.mean([r[metric.replace("raw", "human")] for r in rows]))
            ax.axhline(ref, color="#9AA0A6", lw=1.5, ls=(0, (4, 3)),
                       label="human-human reference" if metric == "mm_raw" else None)
        if normalized:
            ax.axhline(0, color="#9AA0A6", lw=1.2, ls=(0, (4, 3)))
            direction = "higher is better" if higher else "lower is better"
            label = f"Change from each model's first scale ({direction})"
        else:
            label += "  ↑" if higher else "  ↓"
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    for ax in axes[1]:
        ax.set_xlabel("synthetic training images")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(title, y=1.0, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstraps", type=int, default=300)
    args = ap.parse_args()
    entries = json.load(open(args.manifest))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for i, entry in enumerate(entries):
        row = dict(entry)
        row.update(summarize(entry["scores"], args.bootstraps, 3141 + i))
        all_rows.append(row)
    fields = ["domain", "model", "scale", "label", "n_images", "mm_raw", "mm_raw_lo", "mm_raw_hi",
              "mm_human", "mm_kld", "mm_kld_lo", "mm_kld_hi", "sm_raw", "sm_raw_lo", "sm_raw_hi",
              "sm_human", "sm_kld", "sm_kld_lo", "sm_kld_hi"]
    with (out / "sequence_metrics_fixed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    zero = [r for r in all_rows if r["domain"] == "zeroshot"]
    indomain = [r for r in all_rows if r["domain"] == "indomain"]
    plot(zero, out / "zeroshot_mit1003_fixed.png",
         "Zero-shot, length-conditioned sequence similarity on the same 100 MIT1003 images")
    plot(indomain, out / "indomain_sequence_metrics_fixed.png",
         "In-domain generation on the held-out synthetic canonical set")
    plot(indomain, out / "indomain_sequence_scaling_change.png",
         "Within-model in-domain scaling", normalized=True)
    print(f"wrote corrected CSV and plots to {out}")


if __name__ == "__main__":
    main()
