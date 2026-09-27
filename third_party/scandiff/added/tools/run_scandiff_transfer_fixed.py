#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pickle
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
EVAL = ROOT / "data/eval"
OUT = EVAL / "fixed_sequence_eval/scandiff_transfer"
PY = str(ROOT / "../envs/scandiff/bin/python")
COMPUTE = ROOT / "tools/compute_fixed_sequence_scores.py"
BINS = np.linspace(0, 1, 101)


def kld(h, m):
    hp = np.histogram(h, bins=BINS)[0].astype(float); hp /= hp.sum()
    mp = np.histogram(m, bins=BINS)[0].astype(float); mp /= mp.sum()
    return float(entropy(hp, mp + 1e-10))


def aggregate(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    hmm = np.concatenate([v["human_mm"] for v in data.values()])
    mmm = np.concatenate([v["model_mm"] for v in data.values()])
    hsm = np.concatenate([v["human_sm"] for v in data.values()])
    msm = np.concatenate([v["model_sm"] for v in data.values()])
    return dict(mm_raw=mmm.mean(), mm_kld=np.mean([kld(hmm[:, d], mmm[:, d]) for d in range(4)]),
                sm_raw=msm.mean(), sm_kld=kld(hsm, msm), n_images=len(data))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(4):
        human = EVAL / f"mit1003_human_fold{fold}.json"
        for condition in ("scratch", "AV-pretrained"):
            if fold == 0:
                run = ROOT / "runs_ours" / ("train_mit1003_t90" if condition == "scratch" else "ft_pilot_av100k_fold0")
            else:
                run = ROOT / "runs_ours" / (f"train_mit1003_fold{fold}" if condition == "scratch" else f"ft_fold{fold}_av100k")
            epoch = 198 if fold == 0 and condition == "AV-pretrained" else 199
            source = run / f"eval_heldout/generations_epoch_{epoch}/generations_MIT1003Dataset_test.pkl"
            npy = OUT / f"fold{fold}_{condition.lower().replace('-', '_')}.npy"
            scores = OUT / f"fold{fold}_{condition.lower().replace('-', '_')}.pkl"
            if not npy.exists():
                subprocess.run([PY, str(ROOT / "tools/scandiff_gen_to_npy.py"),
                                "--gen-pkl", str(source), "--out", str(npy)], check=True)
            if not scores.exists():
                subprocess.run([PY, str(COMPUTE), "--human", str(human), "--generated", str(npy),
                                "--out", str(scores)], check=True)
            row = {"fold": fold, "condition": condition}
            row.update(aggregate(scores)); rows.append(row)

    fields = ["fold", "condition", "n_images", "mm_raw", "mm_kld", "sm_raw", "sm_kld"]
    with (OUT / "scandiff_transfer_fixed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    metrics = [("mm_raw", "MultiMatch similarity (4D, no duration)", True),
               ("sm_raw", "ScanMatch similarity (no duration)", True),
               ("mm_kld", r"MultiMatch $D_{KL}(P_{HH}\,\Vert\,P_{HM})$", False),
               ("sm_kld", r"ScanMatch $D_{KL}(P_{HH}\,\Vert\,P_{HM})$", False)]
    plt.rcParams.update({"font.size": 11.5, "figure.dpi": 140, "savefig.dpi": 220,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#ECECEC"})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for ax, (metric, title, higher) in zip(axes.flat, metrics):
        vals = []
        for fold in range(4):
            pair = [next(r[metric] for r in rows if r["fold"] == fold and r["condition"] == c)
                    for c in ("scratch", "AV-pretrained")]
            vals.append(pair)
            ax.plot([0, 1], pair, color="#AEB4BA", marker="o", lw=1.2, alpha=.9)
            ax.text(1.025, pair[1], f"F{fold}", fontsize=8, color="#666", va="center")
        mean = np.mean(vals, axis=0)
        ax.plot([0, 1], mean, color="#2e6fbb", marker="o", lw=3, ms=7, label="mean over folds")
        ax.set_xticks([0, 1], ["Scratch", "AV-pretrained"])
        ax.set_title(title + ("  ↑" if higher else "  ↓"))
        ax.margins(x=.15)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("ScanDiff transfer to MIT1003: effect of AV pretraining (4 folds)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "scandiff_transfer_fixed.png", bbox_inches="tight")
    fig.savefig(OUT / "scandiff_transfer_fixed.pdf", bbox_inches="tight")
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
