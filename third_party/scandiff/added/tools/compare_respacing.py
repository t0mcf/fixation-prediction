"""Compare ScanDiff generations at different diffusion-step counts (respacing),
paired on the same 500 images (shard 0). Reports the §4 spatial-collapse
statistics -- the ones that explain why generated data underperforms as
pretraining -- so we can see whether fewer steps change what actually matters.

Reference (respacing 1000): data/scandiffgen_100k/shard_0.parquet
Reduced: data/respace_calib/shard0_respace{50,100,250}.parquet

Usage: python tools/compare_respacing.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

REF = "data/scandiffgen_100k/shard_0.parquet"
CANDS = {1000: REF,
         250: "data/respace_calib/shard0_respace250.parquet",
         100: "data/respace_calib/shard0_respace100.parquet",
         50:  "data/respace_calib/shard0_respace50.parquet"}


def stats(parquet):
    df = pd.read_parquet(parquet)
    xsd, r_abs, sacc, lens = [], [], [], []
    for loc in df["locations"]:
        a = np.asarray(loc, dtype=float).reshape(-1, 2)
        if len(a) < 2:
            continue
        lens.append(len(a))
        xsd.append(a[:, 0])
        r_abs.append(np.sqrt((a ** 2).sum(1)))
        sacc.append(np.sqrt((np.diff(a, axis=0) ** 2).sum(1)))
    x = np.concatenate(xsd); r = np.concatenate(r_abs); s = np.concatenate(sacc)
    return dict(n=len(df), mean_len=np.mean(lens),
                x_sd=x.std(), mean_r=r.mean(), sacc_mean=s.mean())


print(f"{'respacing':>10}{'n':>7}{'mean_len':>10}{'x_sd':>9}{'mean_|r|':>10}{'sacc_mean':>11}")
ref = None
for R, path in CANDS.items():
    if not Path(path).exists():
        print(f"{R:>10}   (missing: {path})")
        continue
    st = stats(path)
    if R == 1000:
        ref = st
    dev = ""
    if ref and R != 1000:
        dr = 100 * (st["mean_r"] - ref["mean_r"]) / ref["mean_r"]
        ds = 100 * (st["sacc_mean"] - ref["sacc_mean"]) / ref["sacc_mean"]
        dev = f"   Δmean_r {dr:+.1f}%  Δsacc {ds:+.1f}%"
    print(f"{R:>10}{st['n']:>7}{st['mean_len']:>10.2f}{st['x_sd']:>9.3f}"
          f"{st['mean_r']:>10.3f}{st['sacc_mean']:>11.3f}{dev}")

print("\nInterpretation: if Δmean_r and Δsacc stay within a few %, the reduced")
print("respacing preserves exactly the geometry that drives the pretraining")
print("result (§4), so it is safe to generate at that step count.")
