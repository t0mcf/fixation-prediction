"""
Paired comparison of the two pretraining sources for the "can a generative model
replace its own training distribution?" experiment: active-vision (the real
source distribution) vs. ScanDiff-generated scanpaths, both fine-tuned on
MIT1003 with the same 10-fold CV split.

Unlike scripts/aggregate_cv.py, which selects the best-val-NSS epoch per fold
(checkpoint-selection leakage: in these CV runs the validation split IS the
held-out fold), this reports both selection rules so the reader can see whether
the conclusion depends on the choice.

The runs aggregated here are the *_final re-fine-tunes (2026-08-12), which start
from each arm's epoch-29 checkpoint. They exist precisely to remove the original
confound: the earlier comparison had selected best_val_ll.pt, which for the
sdgen arm was epoch 0, so it compared 1-epoch-sdgen against 6-epoch-AV.

  python scripts/aggregate_generator_arms.py
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("nss", "ll", "auc")


def fold_rows(log_csv: Path) -> list[dict]:
    """Validation rows of one fold, in file order."""
    if not log_csv.exists():
        return []
    out = []
    with log_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("split") != "val":
                continue
            try:
                out.append({"epoch": float(r["epoch"]),
                            **{m: float(r[m]) for m in METRICS}})
            except (ValueError, KeyError):
                continue
    return out


def per_fold(cond: str, rule: str) -> dict[int, dict]:
    """{fold index -> metrics} under the given checkpoint-selection rule."""
    res = {}
    for fd in sorted((ROOT / "runs" / cond).glob("fold*")):
        rows = fold_rows(fd / "log.csv")
        if not rows:
            continue
        pick = (max(rows, key=lambda d: d["nss"]) if rule == "best-val-nss"
                else max(rows, key=lambda d: d["epoch"]))
        res[int(fd.name.removeprefix("fold"))] = pick
    return res


def paired_t(d: np.ndarray) -> tuple[float, int]:
    """Paired t statistic and degrees of freedom for the differences d."""
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), n - 1
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(n))), n - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ftcv_fair_av100k_final")
    ap.add_argument("--generated", default="ftcv_fair_sdgen100k_final")
    ap.add_argument("--out", default="docs/report_tables/generator_arms.csv")
    args = ap.parse_args()

    rows_out = []
    for rule in ("final-epoch", "best-val-nss"):
        a = per_fold(args.source, rule)
        b = per_fold(args.generated, rule)
        common = sorted(set(a) & set(b))
        if not common:
            raise SystemExit(f"no overlapping folds for rule {rule}: "
                             f"{sorted(a)} vs {sorted(b)}")

        print(f"\n=== checkpoint selection: {rule} "
              f"({len(common)} paired folds: {common}) ===")
        print(f"{'metric':<6} {'active-vision':>20} {'ScanDiff-generated':>22} "
              f"{'diff':>9} {'t':>8} {'wins':>6}")
        for m in METRICS:
            xa = np.array([a[k][m] for k in common])
            xb = np.array([b[k][m] for k in common])
            d = xa - xb
            t, df = paired_t(d)
            wins = int((d > 0).sum())
            print(f"{m:<6} {xa.mean():>11.4f} +/-{xa.std(ddof=1):<6.4f} "
                  f"{xb.mean():>13.4f} +/-{xb.std(ddof=1):<6.4f} "
                  f"{d.mean():>+9.4f} {t:>8.2f} {wins:>3}/{len(common)}")
            rows_out.append({
                "selection": rule, "metric": m, "n_folds": len(common),
                "source_mean": round(float(xa.mean()), 4),
                "source_std": round(float(xa.std(ddof=1)), 4),
                "generated_mean": round(float(xb.mean()), 4),
                "generated_std": round(float(xb.std(ddof=1)), 4),
                "diff": round(float(d.mean()), 4),
                "t": round(t, 3), "df": df, "folds_source_wins": wins,
            })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out}")
    print("Positive diff = the real source distribution beats the generated one.")


if __name__ == "__main__":
    main()
