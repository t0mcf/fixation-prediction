"""Whole-scanpath in-domain results: each CSV row averages the scored sampling
draws within one trained checkpoint. Markers average these per-run means; bars
span training-run means, not the pooled sampling envelope. Selection is unchanged.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_fig_style as tfs
from thesis_fig_style import (COLORS, LABELS, MARKERS, RANGE_KW,
                              apply_style, dodge_x, panel_letter, save,
                              scale_xaxis, series_kw, table)

MODELS = ["ours", "dg3", "scandiff"]
TRUNCATED = {"dg3": {200_000}}


def main():
    apply_style()
    with table("realism_subset_v2.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    # model -> n_images -> [rows of all training runs at that rung]
    by_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[r["model"]][int(r["n_images"])].append(r)
    n_samp = sorted({int(r.get("n_sampling_seeds", 1)) for r in rows})
    n_runs = {m: {x: len(v) for x, v in sorted(d.items())} for m, d in by_model.items()}

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1))
    for k, (ax, name) in enumerate(zip(axes, ("MM", "SM"))):
        col = ("mm_kld", "sm_kld")[k]
        for m in MODELS:
            d = by_model.get(m)
            if not d:
                continue
            xs = sorted(d)
            # marker = mean of per-run sampling means; line = range across runs
            ys = [sum(float(r[col]) for r in d[x]) / len(d[x]) for x in xs]
            lo = [max(0.0, y - min(float(r[col]) for r in d[x]))
                  for y, x in zip(ys, xs)]
            hi = [max(0.0, max(float(r[col]) for r in d[x]) - y)
                  for y, x in zip(ys, xs)]
            ax.errorbar(dodge_x(m, xs), ys, yerr=[lo, hi], **RANGE_KW,
                        **series_kw(m, filled=True), markersize=4.0)
        ax.set_ylabel(f"{name}-KLD ↓")
        ax.set_ylim(bottom=0)
        scale_xaxis(ax)
        ax.set_xlabel("")
        panel_letter(ax, "AB"[k])
    fig.supxlabel("Synthetic training images", y=0.035, fontsize=9)

    def handle_kw(m):
        kw = series_kw(m, filled=True)
        kw["markersize"] = 4.0
        if m == "ours":
            kw["label"] = "Our model"
        return kw
    handles = [Line2D([], [], **handle_kw(m)) for m in MODELS if m in by_model]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0, fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.80,
                        wspace=0.36)

    def first_run(m):
        return [sorted(v, key=lambda r: int(r.get("train_seed", 0)))[0]
                for _, v in sorted(by_model[m].items())] if m in by_model else []
    ckpt = {m: [r["ckpt"] for r in first_run(m)] for m in by_model}
    caption_name = {"ours": "Our model", "dg3": tfs.LABELS["dg3"],
                    "scandiff": tfs.LABELS["scandiff"]}
    runs_txt = "; ".join(
        f"{caption_name[m]} " + ", ".join(f"{x//1000}k: {n}" for x, n in n_runs[m].items())
        for m in MODELS if m in n_runs)
    dg3_txt = (" DeepGaze III uses its best-validation-LL checkpoint; its 200k "
               "rung is the training-budget-limited checkpoint "
               "(2 of 10 stage-2 epochs)." if "dg3" in by_model else
               " DeepGaze III samples are pending.")
    save(fig, "fig2_realism", caption=f"""
Fig 2 — Whole-scanpath distribution fit on the shared 400-image
{tfs.SPLIT} harness: KL divergence between the MultiMatch (A) and ScanMatch
(B) score distributions of generated scanpaths and those of the held-out
active-vision reference (16 reference paths per image; ScanDiff's metric
code; lower is closer, zero would match the reference exactly). The
reference is the synthetic generator's distribution, not human gaze; the
figure therefore measures in-domain generative fidelity rather than
human-likeness. MultiMatch-KLD is the mean of the four duration-free
component divergences (vector, direction, length, position); ScanMatch
uses no duration. One checkpoint per model and training scale, chosen on
validation: our model uses the best-validation-LL checkpoint
({', '.join(sorted(set(ckpt.get('ours', []))))}); ScanDiff uses, per scale, the
checkpoint with the best mean rank of the two divergences over its
every-fifth-epoch validation trajectory ({', '.join(ckpt.get('scandiff', []))}
for {', '.join(r['size'] for r in first_run('scandiff'))}; first training run).{dg3_txt} Markers are means over training runs
(runs per scale: {runs_txt}); each training-run score first averages its {'/'.join(map(str, n_samp))} sampling draws.
The thin vertical lines span the min-max range of these per-run means;
single training runs have no range bar. Markers at the same scale are offset
slightly in x for legibility.""")


if __name__ == "__main__":
    main()
