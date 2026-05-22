#!/bin/bash
#SBATCH --job-name=visualize_models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

python scripts/visualize_predictions.py \
  --checkpoints \
    runs/train_200k_ll_15ep/checkpoints/best_val_nss.pt \
    runs/train_200k_ll_patchpos_15ep/checkpoints/best_val_nss.pt \
    runs/train_200k_visfeat_15ep/checkpoints/best_val_nss.pt \
  --model-names base patchpos visfeat \
  --output-dir runs/viz/comparison_3models \
  --batch-size 64 \
  --num-workers 8 \
  --max-paths-per-image 10 \
  --num-evolution 4 \
  --num-comparison 4 \
  --num-good-bad 8 \
  --scan-batches 50

echo "end: $(date)"
echo "================================================================================"
