#!/bin/bash
#SBATCH --job-name=mk_memmap_100k
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -eo pipefail

SRC=/mnt/lustre-grete/workspaces/ws/nim00018/u27846-nim18-dinov2-features/chunked
OUT=/mnt/lustre-grete/workspaces/ws/nim00018/u27846-nim18-dinov2-features/memmap_oursynth_100k
DATA=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/oursynth_100k
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python

mkdir -p "$OUT"

$PY - "$SRC" "$OUT" "$DATA" "100k" <<'PYEOF'
import sys, json, numpy as np, torch, time, random
from pathlib import Path
from tqdm import tqdm

src_dir  = Path(sys.argv[1])
out_dir  = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
size     = sys.argv[4]

stems = set()
for split in ["train", "validation", "test"]:
    p = data_dir / f"oursynth_{size}_fixations_{split}.json"
    if p.exists():
        with open(p) as f:
            for s in json.load(f):
                stems.add(Path(s["name"]).stem)
print(f"needed stems: {len(stems)}")

with open(src_dir / "index.json") as f:
    chunk_index = json.load(f)

stems_sorted = sorted(s for s in stems if s in chunk_index)
print(f"stems in chunk index: {len(stems_sorted)}")

first_chunk = torch.load(src_dir / chunk_index[stems_sorted[0]], map_location="cpu", weights_only=True)
feat = first_chunk[stems_sorted[0]]
shape = tuple(feat.shape)
dtype = np.float16
print(f"shape: ({len(stems_sorted)}, {shape[0]}, {shape[1]}) dtype: {dtype}")

N = len(stems_sorted)
tmp_npy = out_dir / "features.npy.tmp"
mm = np.memmap(tmp_npy, dtype=dtype, mode="w+", shape=(N, *shape))

stem_to_idx = {}
cache_name = None
cache_data = None
by_chunk = sorted(stems_sorted, key=lambda s: chunk_index[s])

t0 = time.time()
for i, stem in enumerate(tqdm(by_chunk, desc="write")):
    cn = chunk_index[stem]
    if cn != cache_name:
        cache_data = torch.load(src_dir / cn, map_location="cpu", weights_only=True)
        cache_name = cn
    mm[i] = cache_data[stem].numpy().astype(dtype)
    stem_to_idx[stem] = i
mm.flush()
print(f"write seconds: {time.time()-t0:.2f}")

tmp_idx = out_dir / "memmap_index.json.tmp"
with open(tmp_idx, "w") as f:
    json.dump({"shape": [N, *shape], "dtype": "float16", "index": stem_to_idx}, f)

mm_check = np.memmap(tmp_npy, dtype=dtype, mode="r", shape=(N, *shape))
cache_name = None
cache_data = None
rng = random.Random(42)
verify_indices = sorted(rng.sample(range(N), min(50, N)))
max_diff = 0.0
for vi in verify_indices:
    stem = by_chunk[vi]
    cn = chunk_index[stem]
    if cn != cache_name:
        cache_data = torch.load(src_dir / cn, map_location="cpu", weights_only=True)
        cache_name = cn
    orig = cache_data[stem].numpy().astype(dtype)
    d = float(np.max(np.abs(mm_check[vi].astype(np.float32) - orig.astype(np.float32))))
    max_diff = max(max_diff, d)
    print(f"verify {vi} {stem} {cn} max_abs_diff {d}")
print(f"verification max_abs_diff: {max_diff}")
assert max_diff == 0.0, f"verification failed: {max_diff}"

final_npy = out_dir / "features.npy"
final_idx = out_dir / "memmap_index.json"
tmp_npy.rename(final_npy)
tmp_idx.rename(final_idx)
print(f"published: {final_npy} {final_npy.stat().st_size}")
print(f"published: {final_idx} {final_idx.stat().st_size}")
PYEOF
