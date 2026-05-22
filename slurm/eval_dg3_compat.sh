#!/bin/bash
#SBATCH --job-name=eval_dg3_compat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet

set -eo pipefail

mkdir -p logs

echo "================================================================================"
echo "JobID = ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "start: $(date)"
echo "================================================================================"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python scripts/evaluate_dg3_compat.py \
  --checkpoint runs/train_200k_ll_15ep/checkpoints/best_val_nss.pt \
  --batch-size 128 \
  --num-workers 8 \
  --max-images 5000 \
  --max-paths-per-image 10 \
  --eval-batches -1 \
  --eval-resolution 224

echo "end: $(date)"
echo "================================================================================"
