#!/bin/bash
# Score the temperature sweep (10k & 800k x T) into one CSV, then summarize best-T per tag.
# Run after slurm/sample_temp_sweep.sh finishes.
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=$SD/envs_placeholder; PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
HUMAN=$SD/data/eval/mit1003_human_full.json
CSV=${1:-$SD/data/eval/temp_sweep.csv}
rm -f "$CSV"
declare -A N=( [10k]=10000 [800k]=800000 )
for tag in 10k 800k; do
  for T in 0.6 0.8 1.0 1.25 1.5; do
    gen=$SD/data/eval/gen_mit1003_${tag}_T${T}.npy
    [ -f "$gen" ] || { echo "skip ${tag} T${T} (no file)"; continue; }
    echo "=== ${tag} T=${T} ==="
    $PY $SD/tools/score_scanpaths.py --human "$HUMAN" --generated "$gen" \
        --label "ours-${tag}-T${T}" --n-images ${N[$tag]} --temperature $T --csv "$CSV" 2>&1 \
        | grep -E "MultiMatch duration-free|ScanMatch no-dur"
  done
done
echo "=== SUMMARY (best T per checkpoint) ==="
$PY - "$CSV" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for tag,n in [("10k",10000),("800k",800000)]:
    sub=[r for r in rows if r["n_images"]==str(n)]
    if not sub: continue
    best_sim=max(sub,key=lambda r:float(r["mm_sim_model"]))
    best_kld=min(sub,key=lambda r:float(r["mm_kld_nodur"]))
    print(f"{tag}: best MM-sim T={best_sim['temperature']} ({float(best_sim['mm_sim_model']):.3f}) | "
          f"best MM-KLD T={best_kld['temperature']} ({float(best_kld['mm_kld_nodur']):.3f})")
PY
