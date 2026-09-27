"""Headline figure, VARIANT WITH EXTERNAL REFERENCE (2026-09-17; the plain
version is plot_thesis_fig0_headline.py and is not overwritten). Same five
arms, but the metric is IG over the train-only centre-bias prior (LL minus
one constant per split), and a grey dotted line marks the IG that
DeepGaze3.5-VL REPORTS on MIT1003. That number comes from
docs/report_tables/external_references.csv (source + conditions columns),
never from this script; it was obtained under a different split, prior
and resolution and is not on our harness; the caption says so.

Reads fair_fixedsplit_v2.csv (--split test: fair_fixedsplit_v2_test.csv),
224 grid, image-averaged, shared evaluator:
  ours, no pretraining            model=ours    cond=scratch
  ours, 800k synthetic pretraining              cond=bv800k
  DG3, no pretraining             model=dg3_ft  cond=scratch
  DG3, SALICON pretraining                      cond=salicon
  DG3, best synthetic pretraining               the full<size> arm with the
                                                highest mean LL (named in
                                                the caption)
Encoding follows Fig 7: hollow marker = no pretraining, filled =
pretrained; entity colours; marker = mean over runs, thin capless line =
min-max range. Forest layout as Fig 5.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_fig_style as tfs
from thesis_fig_style import (COLORS, MARKERS, RANGE_KW, apply_style, save,
                              table)

RES = 224
METRIC = "ig_cb_img"
EXT_KEY = "dg35_mit_ig"


def load():
    with table("fair_fixedsplit_v2.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["res"]) == RES]
    v = defaultdict(list)
    for r in rows:
        v[(r["model"], r["cond"])].append(float(r[METRIC]))
    dg3_synth = {c: vals for (m, c), vals in v.items()
                 if m == "dg3_ft" and c.startswith("full")}
    best = max(dg3_synth, key=lambda c: sum(dg3_synth[c]) / len(dg3_synth[c]))
    arms = [  # (label, entity, filled, values)
        ("Our model, 800k synthetic pretraining", "ours", True, v[("ours", "bv800k")]),
        ("Our model, no pretraining", "ours", False, v[("ours", "scratch")]),
        ("DeepGaze III, SALICON pretraining", "dg3", True, v[("dg3_ft", "salicon")]),
        (f"DeepGaze III, {best[4:]} synthetic pretraining", "dg3", True, dg3_synth[best]),
        ("DeepGaze III, no pretraining", "dg3", False, v[("dg3_ft", "scratch")]),
    ]
    return arms, best


def load_external():
    with (tfs.RT / "external_references.csv").open() as fh:
        rows = {r["key"]: r for r in csv.DictReader(fh)}
    return rows[EXT_KEY]


def load_cb_reference():
    with (tfs.RT / "centerbias_reference.csv").open() as fh:
        return next(csv.DictReader(fh))


def main():
    apply_style()
    arms, best = load()
    ext = load_external()
    cbr = load_cb_reference()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ranges = []
    ys = list(range(len(arms)))[::-1]
    mean = lambda vals: sum(vals) / len(vals)
    base_of = {ent: mean(vals) for _, ent, filled, vals in arms if not filled}
    for y, (label, ent, filled, vals) in zip(ys, arms):
        m = mean(vals)
        c = COLORS[ent]
        if filled:     # gain over the same model without pretraining
            b = base_of[ent]
            ax.plot([b, m], [y, y], color="#8a8a8a", lw=1.0, zorder=1,
                    solid_capstyle="butt")
            ax.text(m + 0.06, y, f"+{m - b:.2f}", va="center", ha="left",
                    fontsize=7.5, color="#555555")
        ax.errorbar([m], [y], xerr=[[max(0.0, m - min(vals))], [max(0.0, max(vals) - m)]],
                    fmt=MARKERS[ent], ms=7, color=c, mec=c, mew=1.3,
                    mfc=c if filled else "white", **RANGE_KW, zorder=3)
        ranges.append(max(vals) - min(vals))
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in arms])
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.axvline(float(ext["value"]), color="#777777", lw=1.0, ls=":", zorder=1)
    ax.text(float(ext["value"]) + 0.03, (len(arms) - 1) / 2,
            "DeepGaze3.5-VL\nexternal reference", rotation=90,
            ha="left", va="center", fontsize=7, color="#777777", linespacing=1.1)
    ax.set_xlim(0, float(ext["value"]) + 0.30)   # from zero = the centre-bias baseline
    ax.set_xlabel("IG (bits/fixation) ↑")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.44, right=0.98, bottom=0.18, top=0.96)

    # marker diameter in data units (7 pt marker on the rendered axes)
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    width_pt = ax.get_window_extent().width * 72.0 / fig.dpi
    marker_data = 7.0 / width_pt * (x1 - x0)
    range_txt = ("the min-max ranges over runs are narrower than the markers"
                 if max(ranges) < marker_data else
                 "thin lines span the min-max range over runs")
    n_ours = sorted({len(a[3]) for a in arms if a[1] == "ours"})
    n_dg3 = sorted({len(a[3]) for a in arms if a[1] == "dg3"})
    gain_list = [(ent, mean(vals) - base_of[ent]) for _, ent, filled, vals in arms if filled]
    ours_gain = next(g for e, g in gain_list if e == "ours")
    dg3_gains = sorted(round(g, 2) for e, g in gain_list if e == "dg3")
    save(fig, "fig0_headline_sota", caption=f"""
Synthetic pretraining transfers to human gaze. Information gain per
fixation over the center-bias prior (image-averaged, shared 224x224
evaluator) on the MIT1003 {tfs.SPLIT} split after identical fine-tuning on
the MIT1003 training split. Filled markers denote a pretrained
initialization, open markers none (circles: our model, squares: DeepGaze
III); grey bars, labeled with the gain in bits, show each pretrained arm's
gain in IG over the same model's no-pretraining baseline
(+{ours_gain:.2f} for our model; +{' and +'.join(f'{g:.2f}' for g in dg3_gains)} for
DeepGaze III's two pretrained arms). DeepGaze III is
shown with its standard SALICON pretraining and with its best EVALUATED
synthetic-pretraining variant ({best[4:]} images, the highest-IG of the
synthetic arms tested). Markers are means over
{'/'.join(map(str, n_ours))} runs for our model and {'/'.join(map(str, n_dg3))} for DeepGaze III; {range_txt}.
The dotted line is the {ext["value"]} bits that {ext["model"].split(" (")[0]} reports on
MIT1003; that number comes from a different protocol (its own split,
prior and resolution), so it is shown for orientation only and is not
directly comparable to the other points. The axis
starts at zero, the center-bias baseline.""")


if __name__ == "__main__":
    main()
