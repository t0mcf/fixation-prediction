"""Scanpath-length statistics on the MIT1003 cross-family TEST axis (2026-09-22).

Our model and DeepGaze III generate every scanpath with the length of a
reference scanpath of the image; ScanDiff predicts its own length. Both
whole-scanpath metrics compare every generated scanpath with every
reference scanpath of the image (itertools.product in
scandiff/src/gazetools/metrics/{multimatch,scanmatch}.py) and the KLD is
taken against the reference-vs-reference score distribution, so what a
length difference can cost is measured against the length differences
among human viewers. This script reports, per generation set, the mean and
standard deviation of the scanpath length and the mean absolute length
difference over all generated-vs-reference pairs, next to the same
quantity over all reference-vs-reference pairs.

Writes docs/report_tables/length_control_test.csv and
thesis_table_length_control.tex (label tab:length-statistics). Sources:
scandiff/data/eval/mit1003_human_initial_test.json (reference) and
scandiff/data/eval/mit_xfam_test/gen/<label>/gen.npy (generations). The
joint-training AV1k generation comes from the frozen follow-up evaluation under
scandiff/paper_reproduction/joint_av1k_followup_20260923/.
The earlier cut-to-reference-length control (labels *_lenmatch) is
computed but not reported: the scorer pairs every generated with every
reference scanpath, so a per-path length cut is not the quantity the metrics see.
"""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

SD = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval")
EXTERNAL_GENERATIONS = {
    "scandiff_ft_armB_joint_av1k_seed0_bvinit_epoch_170": Path(
        "/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction"
        "/joint_av1k_followup_20260923"
        "/scandiff_ft_armB_joint_av1k_seed0_bvinit_epoch_170/gen.npy"
    ),
}
RT = Path(__file__).resolve().parents[1] / "docs/report_tables"

SETS = [  # (label, family, human data, arm)
    ("ours_ft_scratch_seed42_T1", "Our model", "MIT1003 only", "scratch"),
    ("ours_ft_bv800k_seed42_T1", "Our model", "MIT1003 only", "AV 800k"),
    ("dg3_ft_scratch_rep0_T1", "DeepGaze III", "MIT1003 only", "scratch"),
    ("dg3_ft_full200k_rep0_T1", "DeepGaze III", "MIT1003 only", "AV 200k"),
    ("scandiff_ft_ft_mit_scratch_seed0_epoch_170", "ScanDiff", "MIT1003 only", "scratch"),
    ("scandiff_ft_ft_mit_av1k_seed0_bvinit_epoch_170", "ScanDiff", "MIT1003 only", "AV 1k"),
    ("scandiff_ft_ft_mit_av100k_seed0_bvinit_last", "ScanDiff", "MIT1003 only", "AV 100k"),
    ("scandiff_ft_ft_mit_av200k_seed0_bvinit_last", "ScanDiff", "MIT1003 only", "AV 200k"),
    ("scandiff_ft_r1_official_recipe_seed0_epoch_170", "ScanDiff", "MIT1003 + COCO-FreeView", "scratch"),
    ("scandiff_ft_armB_joint_av1k_seed0_bvinit_epoch_170", "ScanDiff", "MIT1003 + COCO-FreeView", "AV 1k"),
    ("scandiff_released_freeview", "ScanDiff", "released checkpoint", "--"),
]


def main():
    human = {k.lower(): [len(sp["X"]) for sp in v["scanpaths"]] for k, v in json.load((SD / "mit1003_human_initial_test.json").open()).items()}
    ref_all = np.array([L for v in human.values() for L in v])
    ref_pair = np.mean([abs(a - b) for v in human.values() for a, b in itertools.combinations(v, 2)])
    rows = [{"family": "Human reference", "human_data": "--", "arm": "--", "label": "mit1003_human_initial_test",
             "mean_len": round(float(ref_all.mean()), 2), "std_len": round(float(ref_all.std()), 2),
             "mean_abs_len_diff_to_ref": round(float(ref_pair), 2), "n_paths": int(len(ref_all))}]
    for lab, fam, hd, arm in SETS:
        p = EXTERNAL_GENERATIONS.get(
            lab, SD / "mit_xfam_test/gen" / lab / "gen.npy"
        )
        if not p.exists():
            print(f"SKIP {lab}: missing"); continue
        g = np.load(p, allow_pickle=True).item()
        L, d = [], []
        for k, sps in g.items():
            kk = Path(k).stem.lower()
            if kk not in human:
                continue
            Lg = [len(sp) for sp in sps]; L += Lg
            d += [abs(a - b) for a, b in itertools.product(Lg, human[kk])]
        L = np.array(L)
        rows.append({"family": fam, "human_data": hd, "arm": arm, "label": lab,
                     "mean_len": round(float(L.mean()), 2), "std_len": round(float(L.std()), 2),
                     "mean_abs_len_diff_to_ref": round(float(np.mean(d)), 2), "n_paths": int(len(L))})
    with (RT / "length_control_test.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines = [
        "% Scanpath-length statistics on the MIT1003 cross-family test axis (2026-09-22).",
        "% Source: docs/report_tables/length_control_test.csv (scripts/make_length_control_table.py).",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \begin{tabular}{lllrrr}", r"    \toprule",
        r"    Model & Human data & Pretraining & Mean length & SD & Length diff.\ to reference \\",
        r"    \midrule",
    ]
    for r in rows:
        lines.append(f"    {r['family']} & {r['human_data']} & {r['arm']} & {r['mean_len']:.2f} & {r['std_len']:.2f} & {r['mean_abs_len_diff_to_ref']:.2f} \\\\")
    lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{Scanpath lengths on the MIT1003 test set (number of fixations including the initial fixation). "
        r"Our model and DeepGaze~III generate every scanpath with the length of one reference scanpath of the image; "
        r"ScanDiff predicts its own length. The last column is the mean absolute length difference over all "
        r"generated-vs-reference pairs of an image, the pairs that enter MultiMatch and ScanMatch; for the human "
        r"reference it is taken over all reference-vs-reference pairs, which form the comparison distribution of the "
        r"KLD. One training run and one sampling draw per row (the validation-selected checkpoint, seed 42 for our "
        r"model, repetition 0 for DeepGaze~III, seed 0 for ScanDiff).}",
        r"  \label{tab:length-statistics}", r"\end{table}",
    ]
    (RT / "thesis_table_length_control.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
