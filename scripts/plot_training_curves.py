"""
Plot validation training curves for all models.
Reads directly from SLURM log files.
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

# {model_name: {step: {metric: value}}}
logs = {
    "base": """
epoch 001 | step 025000 | val kl=2.4200 | val ll=-6.1955 | val nss=3.8830 | val auc=0.9381
epoch 002 | step 050000 | val kl=2.4070 | val ll=-6.1048 | val nss=4.1250 | val auc=0.9420
epoch 003 | step 075000 | val kl=2.3942 | val ll=-6.0695 | val nss=4.2263 | val auc=0.9436
epoch 004 | step 100000 | val kl=2.4332 | val ll=-6.0553 | val nss=4.2859 | val auc=0.9444
epoch 005 | step 125000 | val kl=2.3945 | val ll=-6.0270 | val nss=4.3534 | val auc=0.9454
epoch 006 | step 150000 | val kl=2.4111 | val ll=-6.0249 | val nss=4.3667 | val auc=0.9456
epoch 007 | step 175000 | val kl=2.3941 | val ll=-6.0051 | val nss=4.4234 | val auc=0.9463
epoch 008 | step 200000 | val kl=2.4028 | val ll=-6.0000 | val nss=4.4284 | val auc=0.9465
epoch 009 | step 225000 | val kl=2.4212 | val ll=-5.9992 | val nss=4.4394 | val auc=0.9468
epoch 010 | step 250000 | val kl=2.4240 | val ll=-5.9901 | val nss=4.4678 | val auc=0.9471
epoch 011 | step 275000 | val kl=2.4149 | val ll=-5.9832 | val nss=4.4828 | val auc=0.9472
epoch 012 | step 300000 | val kl=2.4500 | val ll=-5.9882 | val nss=4.4774 | val auc=0.9473
epoch 013 | step 325000 | val kl=2.4464 | val ll=-5.9860 | val nss=4.4800 | val auc=0.9474
epoch 014 | step 350000 | val kl=2.4524 | val ll=-5.9874 | val nss=4.4794 | val auc=0.9474
epoch 015 | step 375000 | val kl=2.4573 | val ll=-5.9889 | val nss=4.4760 | val auc=0.9474
""",
    "patchpos": """
epoch 001 | step 025000 | val kl=2.4376 | val ll=-6.1750 | val nss=3.9614 | val auc=0.9390
epoch 002 | step 050000 | val kl=2.4061 | val ll=-6.0857 | val nss=4.1909 | val auc=0.9428
epoch 003 | step 075000 | val kl=2.3909 | val ll=-6.0470 | val nss=4.3131 | val auc=0.9444
epoch 004 | step 100000 | val kl=2.4259 | val ll=-6.0303 | val nss=4.3813 | val auc=0.9452
epoch 005 | step 125000 | val kl=2.3956 | val ll=-6.0010 | val nss=4.4467 | val auc=0.9464
epoch 006 | step 150000 | val kl=2.4003 | val ll=-5.9987 | val nss=4.4580 | val auc=0.9465
epoch 007 | step 175000 | val kl=2.3922 | val ll=-5.9870 | val nss=4.4973 | val auc=0.9469
epoch 008 | step 200000 | val kl=2.3967 | val ll=-5.9787 | val nss=4.5156 | val auc=0.9473
epoch 009 | step 225000 | val kl=2.4225 | val ll=-5.9747 | val nss=4.5312 | val auc=0.9476
""",
    "visfeat": """
epoch 001 | step 025000 | val kl=2.3024 | val ll=-6.2744 | val nss=3.7186 | val auc=0.9358
epoch 002 | step 050000 | val kl=2.2452 | val ll=-6.1979 | val nss=3.8954 | val auc=0.9396
epoch 003 | step 075000 | val kl=2.2245 | val ll=-6.1700 | val nss=3.9465 | val auc=0.9412
epoch 004 | step 100000 | val kl=2.2172 | val ll=-6.1496 | val nss=4.0035 | val auc=0.9419
epoch 005 | step 125000 | val kl=2.2035 | val ll=-6.1466 | val nss=4.0166 | val auc=0.9427
epoch 006 | step 150000 | val kl=2.1971 | val ll=-6.1309 | val nss=4.0515 | val auc=0.9430
epoch 007 | step 175000 | val kl=2.1896 | val ll=-6.1253 | val nss=4.0793 | val auc=0.9437
epoch 008 | step 200000 | val kl=2.1849 | val ll=-6.1200 | val nss=4.0877 | val auc=0.9440
epoch 009 | step 225000 | val kl=2.1836 | val ll=-6.1138 | val nss=4.0977 | val auc=0.9440
epoch 010 | step 250000 | val kl=2.1803 | val ll=-6.1071 | val nss=4.1183 | val auc=0.9441
epoch 011 | step 275000 | val kl=2.1781 | val ll=-6.1045 | val nss=4.1234 | val auc=0.9443
""",
}

# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def parse_log(text: str) -> dict:
    pattern = re.compile(
        r"step\s+(\d+)\s+\|.*?val kl=([\d.]+).*?val ll=(-?[\d.]+).*?val nss=([\d.]+).*?val auc=([\d.]+)"
    )
    data = {"step": [], "kl": [], "ll": [], "nss": [], "auc": []}
    for m in pattern.finditer(text):
        data["step"].append(int(m.group(1)))
        data["kl"].append(float(m.group(2)))
        data["ll"].append(float(m.group(3)))
        data["nss"].append(float(m.group(4)))
        data["auc"].append(float(m.group(5)))
    return {k: np.array(v) for k, v in data.items()}

parsed = {name: parse_log(text) for name, text in logs.items()}

# ---------------------------------------------------------------------------
# colours & labels
# ---------------------------------------------------------------------------

colours = {
    "base":     "#2196F3",   # blue
    "patchpos": "#FF9800",   # orange
    "visfeat":  "#4CAF50",   # green
}

display_names = {
    "base":     "Base",
    "patchpos": "Base + PatchPos",
    "visfeat":  "Base + VisFeat",
}

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

metrics = [
    ("ll",  "Val LL (log-prob, ↑ better)",  False),
    ("nss", "Val NSS (↑ better)",           False),
    ("auc", "Val AUC (↑ better)",           False),
    ("kl",  "Val KL (↓ better)",            True),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Validation metrics across training — 200k images, LL loss, cosine LR",
             fontsize=11, y=1.01)

for ax, (metric, ylabel, invert) in zip(axes, metrics):
    for name, d in parsed.items():
        steps_k = d["step"] / 1000
        ax.plot(steps_k, d[metric],
                color=colours[name],
                label=display_names[name],
                linewidth=2,
                marker="o", markersize=4)

    ax.set_xlabel("Training steps (×1000)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if invert:
        ax.invert_yaxis()
    ax.tick_params(labelsize=8)

plt.tight_layout()

out_path = Path("runs/viz/training_curves.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
plt.close()

# ---------------------------------------------------------------------------
# also save individual per-metric figures (cleaner for reading)
# ---------------------------------------------------------------------------

for metric, ylabel, invert in metrics:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in parsed.items():
        steps_k = d["step"] / 1000
        ax.plot(steps_k, d[metric],
                color=colours[name],
                label=display_names[name],
                linewidth=2,
                marker="o", markersize=4)
    ax.set_xlabel("Training steps (×1000)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Validation {metric.upper()} over training", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if invert:
        ax.invert_yaxis()
    plt.tight_layout()
    p = Path(f"runs/viz/training_curve_{metric}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}")
    plt.close()
