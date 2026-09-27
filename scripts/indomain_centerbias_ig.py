#!/usr/bin/env python
"""Centre-bias baseline for the in-domain (synthetic) evaluation, so that
information gain (IG = LL relative to the centre-bias model, Kümmerer et
al. 2015; the convention DeepGaze III reports) can be given next to LL
over uniform, exactly as the MIT1003 fair harness does.

Baseline: one image-independent density fit on active-vision TRAINING
fixations (a random subset of train_800k images, the five training paths
per image, target fixations 1..15 only; the forced centre start is never a
prediction target), with bandwidth and uniform-mixture eps selected by
image-wise cross-validation (src.eval.fair_metrics.fit_centerbias_tuned,
the same routine and candidate grid as on MIT1003). The same map is used
for every rung and both model families; it is fit once per grid.

Output per split (validation_5k / test_10k): per-image mean centre-bias
log-density in bits over uniform, over all 16 paths × 15 targets, at grids
64 and 224 → docs/report_tables/indomain_centerbias_<split>.csv. The
aggregator subtracts it from the per-image LLs of every run:
IG_img = LL_img − CB_img (same images, same targets).

  python scripts/indomain_centerbias_ig.py --n-fit-images 20000
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fair_metrics import fit_centerbias_tuned, coords_to_cells  # noqa: E402

PARQUET = ("/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/"
           "2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet")
SPLITS = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av")
RT = ROOT / "docs/report_tables"


def key(p: str) -> str:
    a = p.strip().split("/"); return a[-2] + "/" + a[-1]


def load_paths(keys: set[str], ids: set[int]) -> dict[str, list[np.ndarray]]:
    """image key -> list of (L,2) arrays for the requested path ids."""
    out: dict[str, list[np.ndarray]] = {}
    pf = pq.ParquetFile(PARQUET)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["image_path", "epoch", "locations"])
        for p, e, loc in zip(t.column("image_path").to_pylist(), t.column("epoch").to_pylist(),
                             t.column("locations").to_pylist()):
            k = key(p)
            if k in keys and int(e) in ids:
                out.setdefault(k, []).append(np.asarray(loc, np.float32).reshape(-1, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit-images", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 224])
    args = ap.parse_args()

    train_keys = [key(l) for l in open(SPLITS / "train_800k.txt") if l.strip()]
    rng = np.random.default_rng(args.seed)
    fit_keys = set(rng.choice(train_keys, size=args.n_fit_images, replace=False).tolist())
    fit_paths = load_paths(fit_keys, set(range(5)))
    assert len(fit_paths) == args.n_fit_images, len(fit_paths)
    per_image_xy = [np.concatenate([sp[1:] for sp in sps]) for sps in fit_paths.values()]
    print(f"fit set: {len(per_image_xy)} train images, {sum(len(x) for x in per_image_xy)} target fixations", flush=True)

    cb = {}
    for g in args.grids:
        cb[g] = fit_centerbias_tuned(per_image_xy, g, verbose=True)     # (g,g) log-density, logsumexp 0
        assert abs(float(torch.logsumexp(cb[g].reshape(-1), 0))) < 1e-4
        np.save(RT / f"indomain_centerbias_map_{g}.npy", cb[g].numpy())

    summary = {"n_fit_images": args.n_fit_images, "seed": args.seed, "path_ids": [0, 1, 2, 3, 4],
               "targets": "fixations 1..15 (centre start excluded)",
               "fit_keys_sha256": hashlib.sha256("\n".join(sorted(fit_keys)).encode()).hexdigest()}
    for split, fname in (("validation", "validation_5k.txt"), ("test", "test_10k.txt")):
        keys = [key(l) for l in open(SPLITS / fname) if l.strip()]
        paths = load_paths(set(keys), set(range(16)))
        assert set(paths) == set(keys), (split, len(paths), len(keys))
        rows = []
        for k in sorted(paths):
            sps = paths[k]; assert len(sps) == 16 and all(len(sp) == 16 for sp in sps), k
            xy = torch.from_numpy(np.concatenate([sp[1:] for sp in sps]))         # (240, 2)
            row = {"image": k.split("/")[-1], "n_fixations": int(len(xy))}
            for g in args.grids:
                r, c = coords_to_cells(xy, g)
                row[f"cb_ll_{g}"] = float(((cb[g][r, c] + math.log(g * g)) / math.log(2.0)).mean())
            rows.append(row)
        out = RT / f"indomain_centerbias_{split}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        means = {g: float(np.mean([r[f"cb_ll_{g}"] for r in rows])) for g in args.grids}
        summary[split] = {"n_images": len(rows), "cb_ll_img_mean": means}
        print(f"{split}: {len(rows)} images | image-averaged centre-bias LL over uniform: "
              + ", ".join(f"{g}: {m:.4f} bits" for g, m in means.items()), flush=True)
    (RT / "indomain_centerbias_summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", RT / "indomain_centerbias_summary.json")


if __name__ == "__main__":
    main()
