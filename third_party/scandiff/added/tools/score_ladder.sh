#!/bin/bash
# Score every available ladder generation into a fresh scaling CSV, then plot.
# Run after slurm/sample_mit1003_ladder.sh (+ the 800k job) have produced the gen npys.
set -e
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
HUMAN=$SD/data/eval/mit1003_human_full.json
CSV=${1:-$SD/data/eval/scaling.csv}
rm -f "$CSV"
declare -A N=( [10k]=10000 [50k]=50000 [200k]=200000 [400k]=400000 [800k]=800000 )
for tag in 10k 50k 200k 400k 800k; do
  gen=$SD/data/eval/gen_mit1003_${tag}.npy
  if [ ! -f "$gen" ]; then echo "skip $tag (no $gen)"; continue; fi
  echo "=== scoring $tag (${N[$tag]} imgs) ==="
  $PY $SD/tools/score_scanpaths.py --human "$HUMAN" --generated "$gen" \
      --label "ours-$tag" --n-images ${N[$tag]} --csv "$CSV" 2>&1 \
      | grep -E "MultiMatch duration-free|ScanMatch no-dur|appended"
done
echo "=== plotting ==="
$PY $SD/tools/plot_scanpath_scaling.py --csv "$CSV"
