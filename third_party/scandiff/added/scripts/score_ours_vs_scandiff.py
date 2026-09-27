"""Score our model's generated scanpaths against the same ground truth ScanDiff
was scored against, using the identical MultiMatch/ScanMatch code, so the two
model numbers are directly comparable. Also computes the human (real-vs-real,
leave-one-out) self-consistency baseline for context.

Usage:
  python scripts/score_ours_vs_scandiff.py \
    --ground-truth-pkl runs_ours/genscore_1k/ground_truth.pkl \
    --our-gen-pkl /path/to/fixation-prediction/runs/fullval_eval/our_gen_1k.pkl \
    --scandiff-results-pkl runs_ours/genscore_1k/scandiff_results.pkl
"""
import argparse
import pickle

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.gazetools.metrics.multimatch import compute_multi_match_score, compute_self_multi_match_score
from src.gazetools.metrics.scanmatch import scan_match_score

IMG_W, IMG_H = 512, 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth-pkl", required=True)
    ap.add_argument("--our-gen-pkl", required=True)
    ap.add_argument("--scandiff-results-pkl", required=True)
    args = ap.parse_args()

    with open(args.ground_truth_pkl, "rb") as f:
        gt = pickle.load(f)
    with open(args.our_gen_pkl, "rb") as f:
        ours = pickle.load(f)
    with open(args.scandiff_results_pkl, "rb") as f:
        scandiff_results = pickle.load(f)

    # only score images both models actually produced (ours skips any with a
    # missing raw ImageNet file, scandiff skips none in this pipeline)
    common_keys = set(gt.keys()) & set(ours.keys())
    print(f"ground truth images: {len(gt)}, our-gen images: {len(ours)}, common: {len(common_keys)}")
    gt_common = {k: gt[k] for k in common_keys}
    ours_common = {k: ours[k] for k in common_keys}

    mm = compute_multi_match_score(gt_common, ours_common, IMG_W, IMG_H)
    mm_means = [sum(x) / len(x) for x in mm]
    ours_mm = sum(mm_means) / len(mm_means)
    print(f"OURS MultiMatch (mean over {len(mm_means)} real/gen pairs): {ours_mm:.4f}")

    sm_scores = scan_match_score(real=gt_common, generated=ours_common, stimulus_width=IMG_W, stimulus_height=IMG_H, tempbin=0)
    ours_sm = sum(sm_scores) / len(sm_scores)
    print(f"OURS ScanMatch (mean over {len(sm_scores)} pairs): {ours_sm:.4f}")

    # human self-consistency baseline (leave-one-out among ground-truth scanpaths)
    human_mm = compute_self_multi_match_score(gt_common, IMG_W, IMG_H)
    human_mm_means = [sum(x) / len(x) for x in human_mm]
    human_mm_score = sum(human_mm_means) / len(human_mm_means)
    print(f"HUMAN (leave-one-out) MultiMatch (mean over {len(human_mm_means)} pairs): {human_mm_score:.4f}")

    print("\n=== SUMMARY ===")
    print(f"{'':12} {'MultiMatch':>12} {'ScanMatch':>12}")
    print(f"{'Human':12} {human_mm_score:>12.4f} {'--':>12}")
    print(f"{'ScanDiff':12} {scandiff_results['multimatch_mean']:>12.4f} {scandiff_results['scanmatch_mean']:>12.4f}")
    print(f"{'Ours':12} {ours_mm:>12.4f} {ours_sm:>12.4f}")


if __name__ == "__main__":
    main()
