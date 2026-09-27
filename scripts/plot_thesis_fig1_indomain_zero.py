"""ALTERNATIVE Fig 1 (2026-09-21; plot_thesis_fig1_indomain.py is not
overwritten): LL and NSS axes start at ZERO (uniform prediction / chance
level); AUC keeps a truncated axis because its chance level is 0.5 and all
values lie in 0.84-0.96.

Thesis Fig 1: in-domain conditional scaling, ours vs DeepGaze III.

Reads: indomain_ours_v2.csv, indomain_dg3_v2.csv, indomain_ceiling.csv.
Selection: best_val for both models (DG3 has no evaluated final epoch;
final.pth is the best weights re-saved).
Conventions (stated in the caption, not on the axes): all three metrics on
the shared 224 grid (decision 2026-09-14: ours lifted from its native 64
grid, DG3 native); columns ll_dg3grid / nss_224 / auc_224 for ours,
ll / nss_224 / auc_224 for DG3. The generator reference is drawn only when
its table for the active split exists (indomain_ceiling.csv is the
validation value; the test value lands as indomain_ceiling_test.csv).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_fig_style as tfs
from thesis_fig_style import (COLORS, RANGE_KW, apply_style, dodge_x,
                              panel_letter, save, scale_xaxis, series_kw,
                              table)


def rows(name, split_dependent=True):
    with table(name, split_dependent).open() as fh:
        return list(csv.DictReader(fh))


def _by_size(rows, key):
    """rows of one model -> (xs, means, [lower_err, upper_err]) over seeds."""
    from collections import defaultdict
    vals = defaultdict(list)
    for r in rows:
        vals[int(r["n_images"])].append(float(r[key]))
    xs = sorted(vals)
    means = [sum(vals[x]) / len(vals[x]) for x in xs]
    err = [[max(0.0, m - min(vals[x])) for m, x in zip(means, xs)],
           [max(0.0, max(vals[x]) - m) for m, x in zip(means, xs)]]
    n_seeds = {x: len(vals[x]) for x in xs}
    return xs, means, err, n_seeds


def main():
    apply_style()
    ours = [r for r in rows("indomain_ours_v2.csv") if r["selection"] == "best_val"]
    dg3 = [r for r in rows("indomain_dg3_v2.csv") if r["selection"] == "best_val"]
    ceil_path = table("indomain_ceiling.csv")
    ceil = rows("indomain_ceiling.csv")[0] if ceil_path.exists() else None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.3, 3.1))

    def ours_plot(ax, key):
        xs, means, err, n_seeds = _by_size(ours, key)
        kw = series_kw("ours", filled=True); kw["markersize"] = 4.0
        ax.errorbar(dodge_x("ours", xs), means, yerr=err, **RANGE_KW, **kw)
        return n_seeds

    TRUNCATED = {200_000}   # DG3 stage 2 stopped at 2/10 epochs

    def dg3_plot(ax, key):
        xs, means, err, n_dg3 = _by_size(dg3, key)
        ax.errorbar(dodge_x("dg3", xs), means, yerr=err, **RANGE_KW,
                    **series_kw("dg3", filled=True), markersize=4.0)
        return n_dg3

    def ceiling_line(ax, key):
        if ceil is not None and ceil.get(key):
            y = float(ceil[key])
            ax.axhline(y, color=COLORS["ref"], lw=1.0, ls=":")
            return y
        return None

    # all three metrics on the shared 224 grid (ours lifted, DG3 native)
    n_seeds = ours_plot(ax1, "ll_dg3grid")
    n_dg3 = dg3_plot(ax1, "ll")
    top = ceiling_line(ax1, "ll_dg3grid")
    ax1.set_ylim(0, top + 0.2 if top else None)
    ax1.set_ylabel("LL (bits/fixation) ↑")
    panel_letter(ax1, "A")

    ours_plot(ax2, "nss_224")
    dg3_plot(ax2, "nss_224")
    top = ceiling_line(ax2, "nss_224")
    ax2.set_ylim(0, top + 0.3 if top else None)
    ax2.set_ylabel("NSS ↑")
    panel_letter(ax2, "B")

    ours_plot(ax3, "auc_224")
    dg3_plot(ax3, "auc_224")
    top = ceiling_line(ax3, "auc_224")
    ax3.set_ylim(0.82, top + 0.01 if top else None)
    ax3.set_ylabel("AUC ↑")
    panel_letter(ax3, "C")

    for ax in (ax1, ax2, ax3):
        scale_xaxis(ax)
        ax.set_xlabel("")
    fig.supxlabel("Synthetic training images", y=0.035, fontsize=9)

    ours_kw = series_kw("ours", filled=True); ours_kw["markersize"] = 4.0
    ours_kw["label"] = "Our model"
    handles = [
        Line2D([], [], **ours_kw),
        Line2D([], [], **series_kw("dg3", filled=True), markersize=4.0),
    ]
    if ceil is not None:
        handles.append(Line2D([], [], color=COLORS["ref"], lw=1.0, ls=":",
                              label="active-vision generator (reference)"))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6,
               handlelength=2.2, fontsize=7.5)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.80,
                        wspace=0.55)
    seeds_txt = ", ".join(f"{k//1000}k: {v}" for k, v in sorted(n_seeds.items()))
    dg3_txt = ", ".join(f"{k//1000}k: {v}" for k, v in sorted(n_dg3.items()))
    n_img = {r["n_img"] for r in ours} | {r["n_img"] for r in dg3}
    n_fix = {r["n_fix"] for r in ours} | {r["n_fix"] for r in dg3}
    assert len(n_img) == 1 and len(n_fix) == 1, (n_img, n_fix)
    ref_txt = ("Dotted lines: the active-vision generator that produced the "
               "corpus, replayed on the same scanpaths and grid."
               if ceil is not None else
               "The generator reference for this split is not yet "
               "available and is omitted.")
    save(fig, "fig1_indomain_zero", caption=f"""
Fig 1 — In-domain scaling of conditional next-fixation prediction on the
held-out synthetic {tfs.SPLIT} set (protocol v2, {int(n_img.pop()):,}
images, all 16 scanpaths per image, {int(n_fix.pop()):,} predicted
fixations). All three metrics are computed on a shared 224x224 grid with
identical target cells for both models: (A) log-likelihood per fixation
relative to a uniform prediction, (B) NSS and (C) AUC. The LL and NSS
axes start at zero (uniform prediction and chance level); the AUC axis
does not (chance level 0.5). Our 64x64 density is
lifted to 224x224 by mass-preserving interpolation, which costs it about
0.02-0.03 bits relative to its native grid; DeepGaze III predicts at 224
natively. {ref_txt} Checkpoints are selected by validation LL for both
models. Markers are means over independent training runs and the thin
vertical lines span the min-max range across those runs; scales with a
single run carry no line (seeds per scale: {seeds_txt}); DeepGaze III runs
per scale: {dg3_txt} (independent repetitions without fixed seeds). Markers at the same scale are offset
slightly in x for legibility.""")

if __name__ == "__main__":
    main()
