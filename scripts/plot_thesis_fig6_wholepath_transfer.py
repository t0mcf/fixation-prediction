"""Thesis Fig 6: whole-scanpath transfer to human gaze vs synthetic scale.

One question: how does each fine-tuned model's whole-scanpath fit to human
MIT1003 scanpaths depend on the amount of synthetic pretraining?

Reads: xfam_mit_v2.csv (--split test: xfam_mit_v2_test.csv), the
cross-family table: every model's fine-tuned checkpoints sampled on the
split's MIT1003 images and scored by one scorer against the human
reference (initial fixation included). Columns used: model, human_data,
cond, n_pretrain, seed, mm_kld (mean of the four duration-free MultiMatch
component KLDs), sm_kld_corr (ScanMatch KLD, width/height-corrected, the
reportable variant on non-square MIT images), n_sampling_seeds.

Series (MIT-only fine-tunes; the joint MIT+COCO arms belong to Fig 7):
  ours          bv<size> (best-val-LL synthetic init), five fine-tuning seeds
  DeepGaze III  full<size> (scanpath-module transfer), reps 0/1; the 200k
                init is budget-limited (open marker, as in Figs 1-3)
  ScanDiff      av_pretrained 1k-100k, seeds 0/1
  dotted line   ScanDiff without synthetic pretraining (light band = range
                over runs). The ours/DG3 scratch lines and the DG3 SALICON
                line are not drawn (the figure became too busy);
                those values are in Fig 3 / the tables.
Markers are means over fine-tuning runs, bars their min-max range,
the same encoding as Fig 3.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_fig_style as tfs
from thesis_fig_style import (COLORS, LABELS, RANGE_KW, apply_style,
                              dodge_x, panel_letter, save, scale_xaxis,
                              series_kw, table)

METRICS = [("mm_kld", "MM-KLD ↓", "A"),
           ("sm_kld_corr", "SM-KLD ↓", "B")]
LADDER = {"ours": "bv", "dg3": "full", "scandiff": "av_pretrained"}
TRUNCATED = {"dg3": {200_000}}


def load():
    with table("xfam_mit_v2.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["human_data"] == "mit_only"]
    # ScanDiff init (column added 2026-09-17): keep the runs fine-tuned from
    # the validation-selected in-domain checkpoint ('selected') and the
    # scratch arms ('-'); drop the earlier final-epoch inits ('last').
    if rows and "init" in rows[0]:
        rows = [r for r in rows if r["model"] != "scandiff"
                or r["init"] in ("selected", "-")]
        sd_init = "selected"
    else:
        sd_init = "last"
    ladder = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base = defaultdict(lambda: defaultdict(list))   # (model, cond) -> metric -> [v]
    for r in rows:
        m, cond = r["model"], r["cond"]
        for key, _, _ in METRICS:
            v = float(r[key])
            if cond == "scratch":
                base[(m, "scratch")][key].append(v)
            elif cond == "salicon":
                base[(m, "salicon")][key].append(v)
            elif (cond.startswith(LADDER[m]) and "lenmatch" not in cond
                  and int(r["n_pretrain"]) > 0):
                ladder[m][key][int(r["n_pretrain"])].append(v)
    n_samp = sorted({int(r["n_sampling_seeds"]) for r in rows})
    return ladder, base, n_samp, sd_init


def _stats(v):
    m = sum(v) / len(v)
    return m, max(0.0, m - min(v)), max(0.0, max(v) - m)


def draw_baseline(ax, entity, values, ls):
    m, lo, hi = _stats(values)
    c = COLORS[entity]
    ax.axhline(m, color=c, lw=0.9, ls=ls, alpha=0.75, zorder=1)
    if hi + lo > 0:
        ax.axhspan(m - lo, m + hi, color=c, alpha=0.10, lw=0, zorder=0)


def draw_ladder(ax, entity, series):
    xs = sorted(series)
    st = [_stats(series[x]) for x in xs]
    ax.errorbar(dodge_x(entity, xs), [s[0] for s in st],
                yerr=[[s[1] for s in st], [s[2] for s in st]],
                **RANGE_KW, **series_kw(entity, filled=True), markersize=4.0)


def main():
    apply_style()
    ladder, base, n_samp, sd_init = load()
    models = [m for m in ("ours", "dg3", "scandiff") if m in ladder]

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1))
    for ax, (key, ylabel, letter) in zip(axes, METRICS):
        if ("scandiff", "scratch") in base:
            draw_baseline(ax, "scandiff", base[("scandiff", "scratch")][key], ":")
        for m in models:
            draw_ladder(ax, m, ladder[m][key])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        scale_xaxis(ax)
        ax.set_xlabel("")
        panel_letter(ax, letter)
    fig.supxlabel("Synthetic pretraining images", y=0.03, fontsize=9)

    display_name = {"ours": "Our model", "dg3": LABELS["dg3"],
                    "scandiff": LABELS["scandiff"]}
    handles = []
    for m in models:
        kw = series_kw(m, filled=True)
        kw["markersize"] = 4.0
        kw["label"] = f"{display_name[m]}, synthetic pretraining"
        handles.append(Line2D([], [], **kw))
    if ("scandiff", "scratch") in base:
        handles.append(Line2D([], [], color=COLORS["scandiff"], lw=0.9, ls=":",
                              alpha=0.75, label="ScanDiff, no pretraining"))
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6,
               handlelength=2.0, handletextpad=0.5, fontsize=7.5)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.77,
                        wspace=0.36)

    def n_runs(m):
        ns = sorted({len(v) for v in ladder[m]["mm_kld"].values()})
        return "/".join(map(str, ns))
    runs_txt = "; ".join(f"{display_name[m]} {n_runs(m)}" for m in models)
    n_sd_scratch = len(base[("scandiff", "scratch")]["mm_kld"]) if ("scandiff", "scratch") in base else 0
    save(fig, "fig6_wholepath_transfer", caption=f"""
Fig 6 — Whole-scanpath transfer to human gaze as a function of synthetic
pretraining. Each model is pretrained on synthetic active-vision ImageNet
scanpaths, fine-tuned on the MIT1003 training split, and its sampled
scanpaths on the MIT1003 {tfs.SPLIT} split are scored against the human
scanpaths of the same images (initial fixation included) with one shared
scorer for all three families. (A) MultiMatch-KLD as the mean of the
four duration-free components (vector, direction, length, position) and
(B) ScanMatch-KLD without duration in the width/height-corrected variant;
lower is closer to the human score distribution. Fine-tuning checkpoints:
Our model and DeepGaze III by validation LL, ScanDiff by the mean validation
rank of the two divergences. Pretraining inits: Our model and DeepGaze III use
their best-validation-LL synthetic checkpoints; ScanDiff uses
{'its validation-selected in-domain checkpoint (the same rule as Fig 2)' if sd_init == 'selected' else 'the final epoch of its in-domain run, unlike Fig 2, which uses the validation-selected checkpoint'}. Markers are means over independent
fine-tuning runs, and the thin vertical lines define the min-max range
across those runs (runs per scale: {runs_txt}; the
DeepGaze III repetitions are not seed-controlled). The dotted line and
shaded band identify ScanDiff fine-tuned on MIT1003 WITHOUT synthetic
pretraining, as the mean and min-max range over
{n_sd_scratch} runs; the corresponding no-pretraining
values of Our model and DeepGaze III, and DeepGaze III's SALICON-pretrained
reference, are omitted here for legibility and reported in Fig 3 and the
tables. DeepGaze III's 200k point starts from the
training-budget-limited in-domain checkpoint. Each training-run score first averages its available sampling draws
({'/'.join(map(str, n_samp))} per checkpoint); the plotted range is across these per-run means. Markers at the same scale are offset
slightly in x for legibility.""")


if __name__ == "__main__":
    main()
