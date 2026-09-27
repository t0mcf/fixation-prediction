#!/bin/bash
#SBATCH --job-name=sd_gen_canon
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet
# ScanDiff generations on the SYNTHETIC canonical val set (in-domain scaling
# axis), so ScanDiff can sit next to ours and DeepGaze III on that axis.
#   sbatch --array=0-1 slurm/gen_canonical_ladder.sh   # 50k, 100k
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
SCALES=(50k 100k)
S=${SCALES[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== canonical-val generation, scandiff-${S} ==="
$PY scripts/gen_and_score_canonical.py --size $S --respacing 1000
echo "DONE $S"
