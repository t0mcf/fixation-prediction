#!/bin/bash
# Convert + score every ScanDiff ladder checkpoint that has a finished
# eval_mit100 generation dir, appending rows to data/eval/scandiff_ladder.csv.
# Idempotent: rebuilds the CSV from scratch each run (rows are cheap).
set -e
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd $SD
CSV=data/eval/scandiff_ladder.csv
rm -f $CSV

declare -A NIMG=( [1k]=1000 [10k_v2]=10000 [50k]=50000 [100k]=100000 [200k]=200000 )
for s in 1k 10k_v2 50k 100k 200k; do
  GLOB="runs_ours/train_${s}/eval_mit100/generations_epoch_*/generations_MIT1003Dataset_test.pkl"
  ls $GLOB >/dev/null 2>&1 || { echo "skip $s (no generations yet)"; continue; }
  NPY=data/eval/gen_scandiff_${s}_m100.npy
  $PY tools/scandiff_gen_to_npy.py --gen-pkl "$GLOB" --out $NPY
  $PY tools/score_scanpaths.py \
    --human data/eval/mit1003_human_m100.json \
    --generated $NPY \
    --label scandiff-${s} --n-images ${NIMG[$s]} \
    --csv $CSV 2>/dev/null | grep -E "^=>"
  # length-matched variant: truncate to index-paired human lengths — the same
  # oracle our sampler gets (no termination head). ScanDiff generates all-16
  # paths because the synthetic training data is 100% fixed-length-16; the
  # as-generated row documents that prior, the lenmatch row is the fair one.
  $PY - <<PYEOF
import numpy as np, json
human = json.load(open('data/eval/mit1003_human_m100.json'))
g = np.load('$NPY', allow_pickle=True).item()
out = {}
for stem, sps in g.items():
    hl = [len(sp['X']) for sp in human[stem]['scanpaths']]
    out[stem] = [np.asarray(sp)[:hl[j % len(hl)]] for j, sp in enumerate(sps)]
np.save('data/eval/gen_scandiff_${s}_m100_lenmatch.npy', out, allow_pickle=True)
PYEOF
  $PY tools/score_scanpaths.py \
    --human data/eval/mit1003_human_m100.json \
    --generated data/eval/gen_scandiff_${s}_m100_lenmatch.npy \
    --label scandiff-${s}-lenmatch --n-images ${NIMG[$s]} \
    --csv $CSV 2>/dev/null | grep -E "^=>"
done
echo "---- $CSV ----"
cat $CSV 2>/dev/null
