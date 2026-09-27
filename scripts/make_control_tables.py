"""CSV sources of two appendix tables that are not produced by
aggregate_v2_results.py.

best_epoch_v2.csv
    Appendix "Validation behavior and additional training of our model":
    epoch of the best validation LL for every synthetic training run of our
    model (standard 18-epoch ladder, the 36-epoch control and the
    horizontal-flip run), read from runs/v2_*/log.csv (rows with
    split == "val"; training-loop evaluator on the native 64x64 grid).

fulltest_control_test.csv
    Appendix "Whole-scanpath evaluation on the full synthetic test set":
    MM-KLD / SM-KLD of the same checkpoint and the same (first) sampling
    draw on the fixed 400-image test subset and on all 10,000 test images,
    for our model (seed 42) and DeepGaze III (first run). Sources: the
    scandiff scoring outputs v2_scoring_subset_test/rung_scores.csv and
    v2_scoring_full_test/rung_scores.csv (MM-KLD = mean of the four
    duration-free components, column mm_kld_nodur; SM-KLD = sm_kld).
    Also prints the Spearman rank correlations and mean differences quoted
    in the appendix text.

    python scripts/make_control_tables.py [--out-dir DIR]
"""
import argparse
import csv
import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SD_EVAL = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval")
SIZES = ["1k", "10k", "50k", "100k", "200k", "400k", "800k"]


def best_epoch_rows():
    runs = glob.glob(str(REPO / "runs/v2_*_5p_ll_seed4*"))
    runs.sort(key=lambda p: (int(os.path.basename(p).split("_")[1].rstrip("k")), p))
    rows = []
    for d in runs:
        name = os.path.basename(d)
        val = [r for r in csv.DictReader(open(f"{d}/log.csv")) if r["split"] == "val"]
        curve = [(int(float(r["epoch"])), float(r["ll"])) for r in val]
        best = max(curve, key=lambda v: v[1])
        variant = "hflip" if "hflip" in name else ("36ep" if "36ep" in name else "standard")
        rows.append({"run": name, "size": name.split("_")[1], "variant": variant, "seed": name[-2:],
                     "epochs": len(curve), "best_epoch": best[0],
                     "best_val_ll_log": round(best[1], 4), "last_val_ll_log": round(curve[-1][1], 4)})
    return rows


def _scores(path):
    return {r["label"]: (float(r["mm_kld_nodur"]), float(r["sm_kld"])) for r in csv.DictReader(open(path))}


def fulltest_rows():
    sub = _scores(SD_EVAL / "v2_scoring_subset_test/rung_scores.csv")
    full = _scores(SD_EVAL / "v2_scoring_full_test/rung_scores.csv")
    rows = []
    for model, fmt in (("ours", "ours_v2_{}_best_val_ll_T1"), ("dg3", "dg3_v2_{}_best_T1")):
        for size in SIZES:
            lab = fmt.format(size)
            if lab not in full:
                continue
            (m4, s4), (m10, s10) = sub[lab], full[lab]
            rows.append({"model": model, "size": size, "mm_400": m4, "mm_10k": m10, "sm_400": s4, "sm_10k": s10})
    return rows


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def write(path, rows, lineterminator="\r\n"):
    # line endings match the files as first written (csv default CRLF for
    # best_epoch_v2.csv, LF for fulltest_control_test.csv), so a re-run is
    # byte-identical to the committed tables
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator=lineterminator)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "docs/report_tables"))
    out = Path(ap.parse_args().out_dir)
    write(out / "best_epoch_v2.csv", best_epoch_rows())
    ft = fulltest_rows()
    write(out / "fulltest_control_test.csv", ft, lineterminator="\n")
    for m in ("mm", "sm"):
        a = [r[f"{m}_400"] for r in ft]
        b = [r[f"{m}_10k"] for r in ft]
        diff = sum(x - y for x, y in zip(a, b)) / len(a)
        print(f"{m.upper()}-KLD over {len(a)} checkpoints: Spearman {spearman(a, b):.3f}, "
              f"mean (400 - 10,000) {diff:.3f}")


if __name__ == "__main__":
    main()
