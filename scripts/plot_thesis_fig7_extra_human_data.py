"""Thesis Fig 7: does the synthetic-pretraining benefit survive more human data?
(redesigned 2026-09-25; the previous LL/SM one-metric-per-model version is
archived as fig7_extra_human_data_prev_llpanel.*)

Two panels, one whole-scanpath metric each, both models in both panels:
  A  MM-KLD  (mean of the four duration-free MultiMatch component KLDs)
  B  SM-KLD  (ScanMatch, no duration, width/height-corrected)
Source: xfam_mit_v2_test.csv (cross-family scorer, same human reference as
Fig 6; test split when the file exists, else validation). Each run's score
is already the mean over its sampling draws (column sampling_aggregation).

x = human training data (MIT1003 = 0, MIT1003 + COCO-FreeView = 1). At each
position one vertical dumbbell per model, offset by DODGE (our model left,
ScanDiff right): hollow marker = no synthetic pretraining, filled =
synthetically pretrained, grey band = change with pretraining (mean to
mean). Markers are means over runs, thin capless lines their min-max range.
Pretraining sizes differ: our model 800k images, ScanDiff 1k images (init =
validation-selected in-domain checkpoint; 'last' only as a named fallback).
Lower KLD is better: a filled marker below its hollow partner means
pretraining helps. DeepGaze III has no joint-data arm (SALICON: Fig 3).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_fig_style import (COLORS, MARKERS, RT, apply_style, panel_letter,
                              save)

XPOS = {"mit_only": 0, "joint": 1}
XLABELS = ["MIT1003", "MIT1003 + COCO-FreeView"]
DODGE = {"ours": -0.10, "scandiff": +0.10}
NAME = {"ours": "Our model", "scandiff": "ScanDiff"}
PRETRAIN_N = {"ours": 800_000, "scandiff": 1_000}
METRICS = [("mm_kld", "MM-KLD ↓", "A"), ("sm_kld_corr", "SM-KLD ↓", "B")]
MS = 6.0            # every regular marker, hollow or filled
EW = 1.3            # regular marker edge width
# Only where a hollow and a filled marker of the same dumbbell nearly coincide
# (vertical distance < one marker) are they nested: hollow 7, filled 5 without
# an edge, filled drawn on top. Marker size otherwise encodes nothing.
MS_NEST_HOLLOW, MS_NEST_FILLED = 7.0, 4.0
PANEL_HEIGHT_PT = 3.2 * 72 * (0.77 - 0.19)   # axes height from figure/subplot settings


def load():
    """(model, arm, human_data) -> metric -> [per-run values]; also the
    split and the cells that fell back from init='selected' to 'last'."""
    test = RT / "xfam_mit_v2_test.csv"
    path, split = (test, "test") if test.exists() else (
        RT / "xfam_mit_v2.csv", "validation")
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["model"] in NAME]
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        m, hd = r["model"], r["human_data"]
        if hd not in XPOS:
            continue
        if r["cond"] == "scratch":
            arm, init = "scratch", "-"
        elif r["cond"] in ("bv800k", "av_pretrained") \
                and int(r["n_pretrain"]) == PRETRAIN_N[m]:
            arm, init = "pretrained", r.get("init", "selected")
        else:
            continue
        for key, _, _ in METRICS:
            cells[(m, arm, hd)][init][key].append(float(r[key]))
    out, fallback = {}, []
    for k, by_init in cells.items():
        if k[1] == "scratch":
            out[k] = by_init["-"]
        elif "selected" in by_init:
            out[k] = by_init["selected"]
        else:
            out[k] = by_init["last"]
            fallback.append(k)
    return out, split, fallback


def _mean(v):
    return sum(v) / len(v)


def draw_panel(ax, cells, key):
    """Returns the list of (model, human_data) dumbbells drawn nested."""
    top = 1.05 * max(max(cell[key]) for cell in cells.values())
    ax.set_ylim(0, top)
    nested = []
    for m in ("ours", "scandiff"):
        c, mk = COLORS[m], MARKERS[m]
        for hd, x0 in XPOS.items():
            x = x0 + DODGE[m]
            sc, pr = cells.get((m, "scratch", hd)), cells.get((m, "pretrained", hd))
            near = False
            if sc and pr:
                ax.plot([x, x], [_mean(sc[key]), _mean(pr[key])],
                        color="#c8c8c8", lw=6, solid_capstyle="butt", zorder=1)
                gap_pt = abs(_mean(sc[key]) - _mean(pr[key])) / top * PANEL_HEIGHT_PT
                near = gap_pt < MS
                if near:
                    nested.append((m, hd))
            # draw order: grey connector (1) < range lines (2) < hollow marker
            # (3) < filled marker (4); range lines and markers are separate
            # artists so a line never shows through a hollow marker.
            for filled, cell in ((False, sc), (True, pr)):
                if not cell:
                    continue
                v = cell[key]
                mu = _mean(v)
                if max(v) > min(v):
                    ax.plot([x, x], [min(v), max(v)], color=c, lw=0.9,
                            solid_capstyle="butt", zorder=2)
                if near:
                    ms = MS_NEST_FILLED if filled else MS_NEST_HOLLOW
                    ew = 0.0 if filled else EW
                else:
                    ms, ew = MS, EW
                ax.plot([x], [mu], ls="none", marker=mk, color=c,
                        markerfacecolor=c if filled else "white",
                        markeredgecolor=c, markeredgewidth=ew, ms=ms,
                        zorder=4 if filled else 3)
    ax.set_xticks(list(XPOS.values()))
    ax.set_xticklabels(XLABELS)
    ax.set_xlim(-0.6, 1.6)
    ax.tick_params(axis="x", length=0)
    return nested


def main():
    apply_style(split="validation")   # file names stay unsuffixed
    cells, split, fallback = load()

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))
    any_nested = False
    for ax, (key, ylabel, letter) in zip(axes, METRICS):
        any_nested |= bool(draw_panel(ax, cells, key))
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter)

    handles = []
    for m in ("ours", "scandiff"):
        c, mk = COLORS[m], MARKERS[m]
        handles.append(Line2D([], [], ls="none", marker=mk, color=c,
                              markerfacecolor="white", markeredgecolor=c,
                              markeredgewidth=EW, ms=MS,
                              label=f"{NAME[m]}, no pretraining"))
        handles.append(Line2D([], [], ls="none", marker=mk, color=c,
                              markerfacecolor=c, markeredgecolor=c,
                              markeredgewidth=EW, ms=MS,
                              label=f"{NAME[m]}, synthetic pretraining"))
    handles.append(Line2D([], [], color="#c8c8c8", lw=6,
                          label="change with synthetic pretraining"))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handlelength=2.0, handletextpad=0.5, fontsize=7.5)
    fig.supxlabel("Human training data", y=0.03, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.77,
                        wspace=0.36)

    nest_txt = (" Nearly coincident markers are overplotted with a smaller "
                "filled marker so that both conditions remain visible."
                if any_nested else "")
    fb_txt = ""
    if fallback:
        fb_txt = (" Exception: for "
                  + ", ".join(f"{NAME[m]} on {XLABELS[XPOS[hd]]}" for m, _, hd in fallback)
                  + " only a run initialized from the final in-domain "
                    "checkpoint exists.")
    counts = "; ".join(
        f"{NAME[m]} " + ", ".join(
            f"{XLABELS[i]} {len(cells[(m, 'scratch', hd)]['mm_kld'])}/"
            f"{len(cells[(m, 'pretrained', hd)]['mm_kld'])}"
            for hd, i in XPOS.items())
        for m in ("ours", "scandiff"))
    save(fig, "fig7_extra_human_data", caption=f"""
Fig 7 — Synthetic pretraining with more human training data. The figure
shows the whole-scanpath metrics, the ones defined for both models; the
complete results, including our model's conditional metrics, are in the
corresponding table. Each model is fine-tuned on MIT1003 alone or on
MIT1003 plus COCO-FreeView, without synthetic pretraining (open markers)
or after it (filled markers); the grey band between them is the change
with synthetic pretraining (mean to mean). Pretraining differs between
the models: our model is pretrained on 800k synthetic images, ScanDiff on
1k. Fine-tuned models are evaluated on the MIT1003 {split} split: sampled
scanpaths are scored against the human scanpaths of the same images with
the shared cross-family scorer of Fig 6; (A) MM-KLD, the mean of the four
duration-free MultiMatch component divergences (vector, direction, length,
position), and (B) SM-KLD, ScanMatch without duration in the
width/height-corrected variant. Lower is better, so a filled marker below
its open partner means pretraining helps. Each run's score first averages
its sampling draws; plotted means and ranges are then computed across runs
(runs per cell, no pretraining/pretrained: {counts}), and the thin lines
span the min-max range; where no line is visible it is narrower than the
marker. Checkpoints: our model by validation LL, ScanDiff by the mean
validation rank of the two divergences, starting from its validation-selected
in-domain checkpoint.{fb_txt} DeepGaze III has no joint-data arm; its
counterpart with additional data, SALICON pretraining, is shown in Fig 3.
Markers of the two models are offset slightly in x for legibility.{nest_txt}""")


if __name__ == "__main__":
    main()
