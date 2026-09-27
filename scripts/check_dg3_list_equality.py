import numpy as np, pandas as pd
from pathlib import Path
PQ="/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
V2=Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av")
df=pd.read_parquet(PQ, columns=["image_path"])
allp=sorted(df["image_path"].unique())
print("images in parquet:", len(allp), flush=True)
rng=np.random.default_rng(3141)
val=[allp[i] for i in rng.permutation(len(allp))[:5000]]
pool=[p for p in allp if p not in set(val)]
perm=rng.permutation(len(pool))
def stems(paths): return [Path(p).stem for p in paths]
def v2list(name): return [Path(l.strip()).stem for l in open(V2/f"{name}.txt") if l.strip()]
print("val == validation_5k:", set(stems(val))==set(v2list("validation_5k")), flush=True)
for N,name in [(1000,"train_1k"),(10000,"train_10k"),(50000,"train_50k"),(100000,"train_100k"),(200000,"train_200k")]:
    tr=[pool[i] for i in perm[:N]]
    ref=v2list(name)
    print(f"{name}: {'MATCH' if set(stems(tr))==set(ref) else 'MISMATCH'} (n={len(ref)})", flush=True)
