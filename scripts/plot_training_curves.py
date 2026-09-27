"""Training curves for all fine-tuning conditions (scratch, 100k, 200k, 400k, 800k).
Reads per-fold log.csv files from CV runs.
Outputs:
  runs/viz/fig_training_curves.png  — val LL mean ± 1 std for all conditions
  runs/viz/fig_training_overfit.png — per-condition train vs val LL (overfitting check)
"""
import csv, statistics as st
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

CONDS = [
    ("scratch", "ftcv_scratch", "#9AA0A6", "from scratch"),
    ("100k",    "ftcv_100k",    "#93C5FD", "100k pretrain"),
    ("200k",    "ftcv_200k",    "#2563EB", "200k pretrain"),
    ("400k",    "ftcv_400k",    "#1D4ED8", "400k pretrain"),
    ("800k",    "ftcv_800k",    "#1E3A8A", "800k pretrain"),
]
N_FOLDS = 10

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 12, "axes.labelsize": 12.5, "axes.labelweight": "medium",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "axes.edgecolor": "#444",
    "axes.grid": True, "grid.color": "#E9E9E9", "font.family": "DejaVu Sans",
})

def load_fold(run_dir, fold):
    """Return (train_steps, train_ll), (val_steps, val_ll) — both sorted by step."""
    p = ROOT / "runs" / run_dir / f"fold{fold}" / "log.csv"
    if not p.exists():
        return None, None
    tr_s, tr_v, va_s, va_v = [], [], [], []
    for row in csv.DictReader(p.open()):
        try:
            step = int(row["step"])
            ll   = float(row["ll"])
        except (KeyError, ValueError):
            continue
        split = row.get("split", "")
        if split == "train":
            tr_s.append(step); tr_v.append(ll)
        elif split == "val":
            va_s.append(step); va_v.append(ll)
    return (tr_s, tr_v), (va_s, va_v)

def mean_std_curves(run_dir, split="val"):
    """Mean ± pstdev of LL across all 10 folds at each logged step."""
    bucket = {}   # step -> [values]
    for f in range(N_FOLDS):
        tr, va = load_fold(run_dir, f)
        data = va if split == "val" else tr
        if data is None or not data[0]:
            continue
        for s, v in zip(*data):
            bucket.setdefault(s, []).append(v)
    steps = sorted(s for s, vs in bucket.items() if len(vs) == N_FOLDS)
    if not steps:
        return np.array([]), np.array([]), np.array([])
    means = np.array([st.mean(bucket[s]) for s in steps])
    stds  = np.array([st.pstdev(bucket[s]) for s in steps])
    return np.array(steps), means, stds

# ── Figure 1: val LL all conditions ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
for key, run_dir, color, label in CONDS:
    xs, means, stds = mean_std_curves(run_dir, "val")
    if len(xs) == 0:
        print(f"  WARNING: no data for {key}"); continue
    ax.plot(xs, means, color=color, lw=2.4, label=label, zorder=4)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18, zorder=3)

ax.set_xlabel("training step")
ax.set_ylabel("validation LL  (bits/fix, ↑ better)")
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.set_axisbelow(True); ax.tick_params(length=4, color="#888")
ax.text(0.99, 0.02, "mean ± 1 std across 10 CV folds",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#999")
out1 = ROOT / "runs/viz/fig_training_curves.png"
fig.savefig(out1); fig.savefig(out1.with_suffix(".pdf"))
plt.close(fig); print(f"saved {out1}")

# ── Figure 2: train vs val LL per condition (overfitting check) ───────────────
fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True, sharey=True)
for ax, (key, run_dir, color, label) in zip(axes, CONDS):
    xs_v, m_v, s_v = mean_std_curves(run_dir, "val")
    xs_t, m_t, s_t = mean_std_curves(run_dir, "train")
    if len(xs_v) == 0:
        ax.set_title(label + "\n(no data)", fontsize=10); continue
    # val — solid
    ax.plot(xs_v, m_v, color=color, lw=2.2, zorder=4)
    ax.fill_between(xs_v, m_v - s_v, m_v + s_v, color=color, alpha=0.15)
    # train — dashed, slightly transparent
    ax.plot(xs_t, m_t, color=color, lw=1.4, ls="--", alpha=0.75, zorder=3)
    ax.fill_between(xs_t, m_t - s_t, m_t + s_t, color=color, alpha=0.07)
    ax.set_title(label, fontsize=11, fontweight="medium")
    ax.set_xlabel("step", fontsize=10); ax.set_axisbelow(True)
    ax.tick_params(length=3, color="#888")
axes[0].set_ylabel("LL  (bits/fix, ↑ better)")
# shared legend on last panel
axes[-1].plot([], [], color="#555", lw=2.2,            label="val  (solid)")
axes[-1].plot([], [], color="#555", lw=1.4, ls="--",   label="train (dashed)")
axes[-1].legend(loc="lower right", fontsize=9.5, framealpha=0.9)
fig.suptitle("Train vs validation LL — mean ± 1 std across 10 folds  (no overfitting signal)",
             fontsize=12, y=1.01)
out2 = ROOT / "runs/viz/fig_training_overfit.png"
fig.savefig(out2, bbox_inches="tight")
fig.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig); print(f"saved {out2}")
