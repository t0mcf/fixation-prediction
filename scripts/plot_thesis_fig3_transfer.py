"""Thesis Fig 3: conditional transfer to human gaze (MIT1003).

Reads: fair_fixedsplit_v2.csv (--split test: fair_fixedsplit_v2_test.csv),
224 grid, image-averaged, shared evaluator, identical target cells for
both families.

Series (all fine-tuned on the MIT1003 training split, best-validation-LL
checkpoint of the fine-tune):
  ours          bv<size> rows: fine-tuned from the best-validation-LL
                synthetic checkpoint (primary init rule since 2026-09-02)
  DeepGaze III  full<size> rows: scanpath-module transfer from the
                synthetic stage; stage-3 checkpoints from the frozen runner
  'none' slot   scratch rows of both families (no synthetic pretraining)
  SALICON line  DG3 fine-tuned from its standard SALICON pretraining
The final-init ladder (lad*), zero-shot rows, the released DG3 weights and
the archived cross-validation numbers are deliberately not drawn.
Markers are means; bars are min–max ranges over seeds (ours) or
repetitions (DG3; the launcher does not set distinct RNG seeds). The DG3
200k init is the training-budget-limited in-domain checkpoint (2/10
stage-2 epochs) and keeps Fig 1's open marker.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_fig_style as tfs
from thesis_fig_style import (COLORS, RANGE_KW, apply_style, dodge_x,
                              panel_letter, save, scale_xaxis, series_kw,
                              table)

RES = 224
TRUNCATED = {200_000}   # DG3 in-domain 200k stopped at 2/10 stage-2 epochs

METRICS = [
    ("ll_img", "LL (bits/fixation)"),
    ("nss_img", "NSS"),
    ("auc_img", "AUC"),
]

# (model column, condition prefix) -> series key
LADDER = {"ours": ("ours", "bv"), "dg3": ("dg3_ft", "full")}
SCRATCH = {"ours": ("ours", "scratch"), "dg3": ("dg3_ft", "scratch")}
SALICON = ("dg3_ft", "salicon")


def load(key):
    with table("fair_fixedsplit_v2.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["res"]) == RES]
    out = {}
    for name, (model, prefix) in LADDER.items():
        vals = defaultdict(list)
        for r in rows:
            if r["model"] == model and r["cond"].startswith(prefix) \
                    and r["cond"][len(prefix):] and r["cond"][len(prefix):][0].isdigit():
                vals[int(r["n_pretrain"])].append(float(r[key]))
        out[name] = dict(sorted(vals.items()))
    for name, (model, cond) in SCRATCH.items():
        out[name + "_scratch"] = [float(r[key]) for r in rows
                                  if r["model"] == model and r["cond"] == cond]
    out["salicon"] = [float(r[key]) for r in rows
                      if r["model"] == SALICON[0] and r["cond"] == SALICON[1]]
    return out


def _stats(values):
    m = sum(values) / len(values)
    return m, max(0.0, m - min(values)), max(0.0, max(values) - m)


def _ci95(values, level=0.95):
    """Mean and two-sided t confidence-interval half-width (n-1 degrees of
    freedom); a single run has no interval."""
    from math import sqrt
    from statistics import stdev
    from scipy.stats import t
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0, 0.0
    half = t.ppf(0.5 + level / 2, n - 1) * stdev(values) / sqrt(n)
    return m, half, half


def _std(values):
    """Mean and +-1 sample standard deviation (n-1 denominator); a single
    run has no spread."""
    from statistics import stdev
    n = len(values)
    m = sum(values) / n
    sd = stdev(values) if n > 1 else 0.0
    return m, sd, sd


def draw_ladder(ax, entity, series, filled=False, markersize=None, stat=_stats):
    xs = list(series)
    stats = [stat(series[x]) for x in xs]
    kw = series_kw(entity, filled=filled)
    if markersize:
        kw["markersize"] = markersize
    ax.errorbar(dodge_x(entity, xs), [s[0] for s in stats],
                yerr=[[s[1] for s in stats], [s[2] for s in stats]],
                **RANGE_KW, **kw)


def draw_baseline(ax, entity, values, ls, stat=_stats):
    """Horizontal reference: mean as a line, interval (min-max range by
    default) as a light band."""
    m, lo, hi = stat(values)
    c = COLORS[entity]
    ax.axhline(m, color=c, lw=0.9, ls=ls, alpha=0.75, zorder=1)
    if hi + lo > 0:
        ax.axhspan(m - lo, m + hi, color=c, alpha=0.10, lw=0, zorder=0)


METRIC_TEXT = {
    "ll_img": "LL relative to a uniform prediction",
    "ig_cb_img": "information gain relative to the train-only centre-bias "
                 "prior (a KDE over the training-split fixations, identical "
                 "for both families)",
    "nss_img": "NSS",
    "auc_img": "AUC",
}


def main(metrics=METRICS, name="fig3_transfer_metrics", ours_label="ours",
        legend_ncol=None, legend_model_columns=False, filled_markers=False,
        marker_size=None, interval="range", extra_caption=""):
    apply_style()
    data = {key: load(key) for key, _ in metrics}
    stat = {"range": _stats, "ci95": _ci95, "std": _std}[interval]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.3 if n > 1 else 4.4, 3.2))
    axes = [axes] if n == 1 else list(axes)
    for ax, (key, ylabel), letter in zip(axes, metrics, "ABC"):
        d = data[key]
        draw_baseline(ax, "ours", d["ours_scratch"], ":", stat=stat)
        draw_baseline(ax, "dg3", d["dg3_scratch"], ":", stat=stat)
        draw_baseline(ax, "dg3", d["salicon"], "-.", stat=stat)
        draw_ladder(ax, "ours", d["ours"], filled=filled_markers, markersize=marker_size, stat=stat)
        draw_ladder(ax, "dg3", d["dg3"], filled=filled_markers, markersize=marker_size, stat=stat)
        ax.set_ylabel(ylabel)
        scale_xaxis(ax)
        ax.set_xlabel("")
        if n > 1:
            panel_letter(ax, letter)
    fig.supxlabel("Synthetic pretraining images", y=0.03, fontsize=9)

    ours_kw = series_kw("ours", filled=filled_markers)
    if marker_size:
        ours_kw["markersize"] = marker_size
    ours_kw["label"] = f"{ours_label}, synthetic pretraining"
    dg3_kw = series_kw("dg3", filled=filled_markers)
    if marker_size:
        dg3_kw["markersize"] = marker_size
    dg3_kw["label"] = "DeepGaze III, synthetic pretraining"
    handles = [
        Line2D([], [], **ours_kw),
        Line2D([], [], color=COLORS["ours"], lw=0.9, ls=":", alpha=0.75,
               label=f"{ours_label}, no pretraining"),
        Line2D([], [], **dg3_kw),
        Line2D([], [], color=COLORS["dg3"], lw=0.9, ls=":", alpha=0.75,
               label="DeepGaze III, no pretraining"),
        Line2D([], [], color=COLORS["dg3"], lw=0.9, ls="-.", alpha=0.75,
               label="DeepGaze III, SALICON pretraining"),
    ]
    if legend_model_columns:
        # two columns grouped by model: an invisible spacer after the two
        # "ours" entries pushes the three DeepGaze III entries into column 2
        # (matplotlib fills legend columns top to bottom)
        handles.insert(2, Line2D([], [], color="none", label=" "))
        legend_ncol = 2
    fig.legend(handles=handles, loc="upper center",
               ncol=legend_ncol if legend_ncol is not None else (3 if n > 1 else 2),
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
               handlelength=1.8, handletextpad=0.4, fontsize=7.5)
    fig.subplots_adjust(left={3: 0.085, 2: 0.10, 1: 0.14}[n], right=0.99,
                        bottom=0.19, top=0.76 if n > 1 else 0.74,
                        wspace=0.55 if n == 3 else 0.36)

    # seed / repetition counts for the caption, from the rows actually drawn
    ll = data[metrics[0][0]]
    panels = ("; ".join(f"({L}) {METRIC_TEXT[k]}" for L, (k, _) in zip("ABC", metrics))
              if n > 1 else METRIC_TEXT[metrics[0][0]])
    ours_n = {x: len(v) for x, v in ll["ours"].items()}
    dg3_n = {x: len(v) for x, v in ll["dg3"].items()}
    ours_txt = ", ".join(f"{x//1000}k: {n}" for x, n in ours_n.items())
    dg3_set = sorted(set(dg3_n.values()))
    dg3_txt = (f"{dg3_set[0]} repetitions per scale" if len(dg3_set) == 1
               else ", ".join(f"{x//1000}k: {n}" for x, n in dg3_n.items()))
    band_txt = "(light bands: their range over runs)"
    bar_txt = ("the thin vertical lines span the\n"
               "min-max range over independent fine-tuning runs")
    if interval == "ci95":
        band_txt = "(light bands: their 95% confidence interval of the mean)"
        bar_txt = ("the thin vertical lines show the 95% confidence interval "
                   "of the mean\n(two-sided t interval, n-1 degrees of freedom) "
                   "over independent fine-tuning runs")
    if interval == "std":
        band_txt = "(light bands: their mean \u00b11 standard deviation)"
        bar_txt = ("the thin vertical lines span the mean \u00b11 sample standard "
                   "deviation\n(n-1 denominator) over independent fine-tuning runs")
    save(fig, name, caption=f"""
Fig 3 — Transfer of synthetic pretraining to human gaze. Every model is
fine-tuned on the MIT1003 training split (702 images) with identical
hyperparameters and evaluated on the {tfs.SPLIT} split under the shared
224x224 evaluator, image-averaged, both families scored at identical
target cells: {panels}. The x axis gives
the number of synthetic active-vision images seen before fine-tuning;
dotted lines are training from scratch on MIT1003 alone
{band_txt}. Our model initialises
from the best-validation-LL synthetic checkpoint; DeepGaze III transfers
its scanpath module from the synthetic stage and its 200k point starts
from the training-budget-limited in-domain checkpoint (2 of 10 stage-2
epochs). The dash-dotted line is DeepGaze III fine-tuned from its
standard SALICON pretraining. Fine-tuning checkpoints are selected by
validation LL. Markers are means and {bar_txt}, which share one pretraining checkpoint per
scale: {ours_label} {ours_txt} training seeds (scratch: {len(ll['ours_scratch'])});
DeepGaze III {dg3_txt} (scratch: {len(ll['dg3_scratch'])}, SALICON:
{len(ll['salicon'])}); the DeepGaze III launcher does not set distinct
RNG seeds, so its repetitions are not seed-controlled. Markers at the same scale are offset
slightly in x for legibility.{extra_caption}""")


if __name__ == "__main__":
    main()
