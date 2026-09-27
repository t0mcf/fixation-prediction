#!/usr/bin/env python
"""Thesis tables (LaTeX, booktabs) from the TEST report tables (2026-09-17).

Every number is read from docs/report_tables/*_test.csv (never typed in).
Means over training seeds/repetitions; the seed range is given where more
than one repetition exists. Whole-scanpath scores average sampling draws within each training run first.

  python scripts/make_thesis_tables_v2.py            -> docs/report_tables/thesis_tables_test.tex
  python scripts/make_thesis_tables_v2.py --split validation
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd

RT = Path(__file__).resolve().parents[1] / "docs/report_tables"
SIZES = [1000, 10000, 50000, 100000, 200000, 400000, 800000]
SZ = {1000: "1k", 10000: "10k", 50000: "50k", 100000: "100k", 200000: "200k", 400000: "400k", 800000: "800k"}


def f3(x, nd=3):
    return "--" if x is None or pd.isna(x) else f"{x:.{nd}f}"


RANGES = False   # --ranges: append [min, max] over repetitions (appendix variant)


def mean_range(vals, nd=3):
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return "--"
    m = sum(vals) / len(vals)
    if len(vals) == 1 or not RANGES:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} \\scriptsize[{min(vals):.{nd}f}, {max(vals):.{nd}f}]"


def block(caption, label, cols, header, rows, note=""):
    body = "\n".join(rows)
    return (f"\\begin{{table}}[t]\n  \\centering\n  \\small\n  \\begin{{tabular}}{{{cols}}}\n    \\toprule\n"
            f"    {header} \\\\\n    \\midrule\n{body}\n    \\bottomrule\n  \\end{{tabular}}\n"
            f"  \\caption{{{caption.strip()}{(' ' + note.strip()) if note else ''}}}\n  \\label{{{label}}}\n\\end{{table}}\n")


def tab_indomain(sfx):
    ours = pd.read_csv(RT / f"indomain_ours_v2{sfx}.csv")
    dg3_all = pd.read_csv(RT / f"indomain_dg3_v2{sfx}.csv")
    dg3_runs = dg3_all.groupby("n_images").size()          # training runs per scale (rep1 at 1k/10k)
    dg3 = dg3_all.groupby("n_images")[["ll", "nss_224", "auc_224"]].mean()
    ceil = pd.read_csv(RT / f"indomain_ceiling{sfx}.csv").iloc[0]
    rows = []
    for n in SIZES:
        o = ours[ours.n_images == n]
        d = dg3.loc[n] if n in dg3.index else None
        rows.append("    " + " & ".join([
            SZ[n] + ("$^{*}$" if len(o) == 1 else ""),
            # IG is not reported in-domain (decided 2026-09-20): it equals LL minus a constant and has no
            # published counterpart on synthetic data. The ig_224 columns stay in the CSVs.
            mean_range(o.ll_dg3grid), mean_range(o.nss_224), mean_range(o.auc_224),
            f3(d.ll) if d is not None else "--",
            f3(d.nss_224) if d is not None else "--", f3(d.auc_224) if d is not None else "--",
        ]) + (" \\\\" if n != 200000 else " \\\\[1pt]"))
    rows.append("    \\midrule")
    rows.append(f"    AV generator & {f3(ceil.ll_dg3grid)} & {f3(ceil.nss_224)} & {f3(ceil.auc_224)} & \\multicolumn{{3}}{{c}}{{}} \\\\")
    header = ("& \\multicolumn{3}{c}{ours} & \\multicolumn{3}{c}{DeepGaze III} \\\\\n"
              "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n    Images & LL & NSS & AUC & LL & NSS & AUC")
    return block(
        "In-domain conditional prediction on the synthetic test split (10,000 images, all 16 scanpaths per image, "
        "2.4 million predicted fixations) as a function of training-set size. All metrics on the shared "
        "$224\\times224$ grid, image-averaged; LL in bits per fixation relative to a uniform prediction "
        "(Section~\\ref{evaluation-conditional}). Checkpoints selected by validation LL. "
        "Our values are means over three training seeds up to 50k and two at 100k and 200k (the seed range never exceeds 0.02 bits LL); rows marked $^{*}$ are single runs. DeepGaze III values are means over two training runs at 1k and 10k and single runs above, "
        "and its 200k run completed 2 of 10 scanpath-stage epochs. The last row is the generating active-vision model "
        "scored on the same fixations.",
        "tab:indomain", "lrrrrrr", header, rows)


def tab_realism(sfx):
    """Average sampling draws within runs, then report means/ranges across runs."""
    df = pd.read_csv(RT / f"realism_subset_v2{sfx}.csv")
    ntrain = {}
    def cell(m, n, k):
        r = df[(df.model == m) & (df.n_images == n)]
        if r.empty:
            return "--"
        ntrain.setdefault(m, set()).add(len(r))
        return mean_range(r[k])
    rows = ["    " + " & ".join([SZ[n]] + [cell(m, n, k) for m in ("ours", "dg3", "scandiff") for k in ("mm_kld", "sm_kld")]) + " \\\\" for n in SIZES]
    def runs_txt(m):
        # exact statement per model: which scales have how many training runs
        r = df[df.model == m].groupby("n_images").size()
        if r.empty or (r == 1).all():
            return "one training run per scale"
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
        parts = []
        for k in sorted(r.unique(), reverse=True):
            sizes = [SZ[n] for n in SIZES if n in r.index and r[n] == k]
            parts.append(f"{words.get(k, k)} at " + ", ".join(sizes))
        return "; ".join(parts)
    header = ("& \\multicolumn{2}{c}{ours} & \\multicolumn{2}{c}{DeepGaze III} & \\multicolumn{2}{c}{ScanDiff} \\\\\n"
              "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
              "    Images & MM-KLD & SM-KLD & MM-KLD & SM-KLD & MM-KLD & SM-KLD")
    return block(
        "In-domain whole-scanpath similarity on the fixed 400-image test subset: KL divergence between the score "
        "distributions of generated and active-vision reference scanpaths (duration-free MultiMatch and ScanMatch, "
        "lower is better). Five generated scanpaths per image; our model and DeepGaze III sampled at temperature 1 with "
        "the reference lengths, ScanDiff with its own lengths. Checkpoints selected on validation (ours and DeepGaze III by "
        "LL, ScanDiff by the mean rank of the two divergences). Values are means over training runs "
        f"(ours: {runs_txt('ours')}; DeepGaze III: {runs_txt('dg3')}; ScanDiff: {runs_txt('scandiff')}). "
        "The two sampling-draw scores are averaged within each training run before averaging across runs. "
        + ("Brackets show the min--max range of the per-run means; single runs have no range." if RANGES else
           "Ranges across training-run means are provided in the companion range table."),
        "tab:realism", "lrrrrrr", header, rows)


def tab_transfer(sfx):
    df = pd.read_csv(RT / f"fair_fixedsplit_v2{sfx}.csv")
    df = df[df.res == 224]
    def line(label, model, cond, indent=""):
        r = df[(df.model == model) & (df.cond == cond)]
        if r.empty:
            return None
        return "    " + " & ".join([indent + label, mean_range(r.ll_img), mean_range(r.ig_cb_img),
                                    mean_range(r.nss_img), mean_range(r.auc_img, 4)]) + " \\\\"
    rows = ["    \\multicolumn{5}{l}{\\emph{ours}} \\\\", line("no synthetic pretraining", "ours", "scratch")]
    rows += [line(f"AV pretraining, {SZ[n]}", "ours", f"bv{SZ[n]}") for n in SIZES]
    rows += ["    \\midrule", "    \\multicolumn{5}{l}{\\emph{DeepGaze III}} \\\\", line("no synthetic pretraining", "dg3_ft", "scratch"),
             line("SALICON pretraining", "dg3_ft", "salicon")]
    rows += [line(f"AV pretraining, {SZ[n]}", "dg3_ft", f"full{SZ[n]}") for n in SIZES[:5]]
    rows += ["    \\midrule", line("released weights (contaminated)", "dg3_released", "released")]
    rows = [r for r in rows if r]
    header = "Model / pretraining & LL & IG & NSS & AUC"
    return block(
        "Conditional prediction of human gaze on the MIT1003 test split (151 images, 15,588 fixations) after "
        "identical human-gaze training, as a function of synthetic pretraining. Shared $224\\times224$ grid, "
        "image-averaged; IG relative to a centre bias fit on the training split. Checkpoints selected by validation LL. "
        "Our model: means over five fine-tuning seeds at every row; DeepGaze III: two independent training repetitions; released weights: one evaluation (ranges in Appendix~\\ref{app:tables}). "
        "The released DeepGaze III weights were trained on all of MIT1003 including this test split and are a reference only.",
        "tab:transfer", "lrrrr", header, rows)


def tab_additional(sfx):
    df = pd.read_csv(RT / f"fair_fixedsplit_v2{sfx}.csv"); df = df[(df.res == 224) & (df.model == "ours")]
    sd = pd.read_csv(RT / f"scandiff_ft_v2{sfx}.csv")
    xf = pd.read_csv(RT / f"xfam_mit_v2{sfx}.csv")
    rows = ["    \\multicolumn{5}{l}{\\emph{ours, conditional (LL / IG)}} \\\\"]
    for hd, conds in (("MIT1003", ("scratch", "bv800k")), ("MIT1003 + COCO-FreeView", ("joint_scratch", "joint_bv800k"))):
        r0, r1 = df[df.cond == conds[0]], df[df.cond == conds[1]]
        rows.append("    " + " & ".join([hd, mean_range(r0.ll_img), mean_range(r1.ll_img), mean_range(r0.ig_cb_img), mean_range(r1.ig_cb_img)]) + " \\\\")
    rows.append("    \\midrule")
    rows.append("    \\multicolumn{5}{l}{\\emph{ours, whole-scanpath (MM-KLD / SM-KLD)}} \\\\")
    for hd, key in (("MIT1003", "mit_only"), ("MIT1003 + COCO-FreeView", "joint")):
        r0 = xf[(xf.model == "ours") & (xf.human_data == key) & (xf.cond == "scratch")]
        r1 = xf[(xf.model == "ours") & (xf.human_data == key) & (xf.cond == "bv800k")]
        rows.append("    " + " & ".join([hd, mean_range(r0.mm_kld), mean_range(r1.mm_kld), mean_range(r0.sm_kld_corr), mean_range(r1.sm_kld_corr)]) + " \\\\")
    rows.append("    \\midrule")
    rows.append("    \\multicolumn{5}{l}{\\emph{ScanDiff, whole-scanpath (MM-KLD / SM-KLD)}} \\\\")
    for hd, key in (("MIT1003", "mit_only"), ("MIT1003 + COCO-FreeView", "joint")):
        # 2026-09-19: same scorer and reference as the ours rows (cross-family axis), and the
        # fine-tunes that start from the validation-SELECTED synthetic checkpoint are primary
        sx = xf[(xf.model == "scandiff") & (xf.human_data == key)]
        r0 = sx[sx.cond == "scratch"]
        # pretrained arm: the scale selected on MIT1003 validation (1k), for both human-data settings
        r1 = sx[(sx.cond == "av_pretrained") & (sx.n_pretrain == 1000) & (sx.init == "selected")]
        rows.append("    " + " & ".join([hd, mean_range(r0.mm_kld), mean_range(r1.mm_kld), mean_range(r0.sm_kld_corr), mean_range(r1.sm_kld_corr)]) + " \\\\")
    header = ("& \\multicolumn{2}{c}{first metric} & \\multicolumn{2}{c}{second metric} \\\\\n"
              "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n    Human training data & scratch & AV pretrained & scratch & AV pretrained")
    return block(
        "Effect of additional human training data on the MIT1003 test split. For each model, training without synthetic "
        "pretraining (scratch) is compared with training initialised from active-vision pretraining (ours: 800,000 images; "
        "ScanDiff: 1,000 images, the scales selected on MIT1003 validation), on MIT1003 alone and on MIT1003 together with COCO-FreeView. Our model: LL and IG "
        "(conditional) and duration-free MultiMatch-/ScanMatch-KLD against the human scanpaths (whole-scanpath); ScanDiff: "
        "whole-scanpath only. Whole-scanpath scores average the two sampling draws within each run before averaging across runs. Our MIT1003-only results use five training runs; all other entries use two (ranges in Appendix~\\ref{app:tables}).",
        "tab:additional-data", "lrrrr", header, rows)


def tab_xfam(sfx):
    xf = pd.read_csv(RT / f"xfam_mit_v2{sfx}.csv")
    def line(label, model, hd, cond, n=None):
        r = xf[(xf.model == model) & (xf.human_data == hd) & (xf.cond == cond)]
        if n is not None:
            r = r[r.n_pretrain == n]
        if model == "scandiff" and cond == "av_pretrained":
            r = r[r.init == "selected"]      # primary arm since 2026-09-19; last-init rows stay in the CSV
        if r.empty:
            return None
        return "    " + " & ".join([label, mean_range(r.mm_kld), mean_range(r.sm_kld_corr)]) + " \\\\"
    rows = ["    \\multicolumn{3}{l}{\\emph{ours}} \\\\", line("no synthetic pretraining", "ours", "mit_only", "scratch")]
    rows += [line(f"AV pretraining, {SZ[n]}", "ours", "mit_only", f"bv{SZ[n]}") for n in SIZES]
    rows += ["    \\midrule", "    \\multicolumn{3}{l}{\\emph{DeepGaze III}} \\\\", line("no synthetic pretraining", "dg3", "mit_only", "scratch"),
             line("SALICON pretraining", "dg3", "mit_only", "salicon")]
    rows += [line(f"AV pretraining, {SZ[n]}", "dg3", "mit_only", f"full{SZ[n]}") for n in SIZES[:5]]
    rows += ["    \\midrule", "    \\multicolumn{3}{l}{\\emph{ScanDiff}} \\\\", line("no synthetic pretraining", "scandiff", "mit_only", "scratch")]
    rows += [line(f"AV pretraining, {SZ[n]}", "scandiff", "mit_only", "av_pretrained", n) for n in SIZES[:5]]
    rows += ["    \\midrule", line("released checkpoint (MIT1003 + COCO)", "scandiff", "joint", "released")]
    rows = [r for r in rows if r]
    header = "Model / pretraining & MM-KLD & SM-KLD"
    return block(
        "Whole-scanpath similarity to human gaze on the MIT1003 test split after human-gaze training on MIT1003: KL "
        "divergence between the score distributions of generated and human scanpaths (duration-free MultiMatch; ScanMatch "
        "with the width/height-corrected grid; lower is better). One generated scanpath per human scanpath, including the "
        "initial central fixation; our model and DeepGaze III are length-matched to the human scanpaths and sampled at "
        "temperature 1, ScanDiff determines its own lengths. Human-gaze training of pretrained models starts from the validation-selected synthetic checkpoint. Sampling-draw scores are averaged within each training run, then across training runs. Ranges, where shown, span these per-run means; "
        "the released ScanDiff checkpoint is a reference row.",
        "tab:xfam", "lrr", header, rows)


def tab_triangle(sfx):
    df = pd.read_csv(RT / f"fair_fixedsplit_v2{sfx}.csv"); df = df[(df.res == 224) & (df.model == "ours")]
    rows = []
    for label, cond in (("no synthetic pretraining", "scratch"), ("active vision, full length", "avfull"),
                        ("active vision, length-matched", "avmatch"), ("ScanDiff-generated", "sd5")):
        r = df[df.cond == cond]
        rows.append("    " + " & ".join([label, mean_range(r.ll_img), mean_range(r.ig_cb_img), mean_range(r.nss_img), mean_range(r.auc_img, 4)]) + " \\\\")
    header = "Pretraining corpus & LL & IG & NSS & AUC"
    return block(
        "Source comparison (appendix): MIT1003 test performance of our model after identical fine-tuning from three "
        "pretraining corpora on the same 100,000 ImageNet images (30 pretraining epochs each; scratch reused from the main "
        "experiment). Shared $224\\times224$ grid, image-averaged; means over three fine-tuning seeds, five for the no-pretraining row (ranges in Appendix~\\ref{app:tables}).",
        "tab:triangle", "lrrrr", header, rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--ranges", action="store_true", help="append [min, max] over repetitions (appendix tables)")
    a = ap.parse_args(); sfx = "" if a.split == "validation" else "_test"
    global RANGES; RANGES = a.ranges
    parts = [tab_indomain(sfx), tab_realism(sfx), tab_transfer(sfx), tab_xfam(sfx), tab_additional(sfx), tab_triangle(sfx)]
    out = RT / f"thesis_tables_{a.split}{'_ranges' if a.ranges else ''}.tex"
    out.write_text(f"% generated by scripts/make_thesis_tables_v2.py --split {a.split}; numbers from docs/report_tables/*{sfx}.csv\n\n" + "\n".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main()
