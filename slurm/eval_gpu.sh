#!/bin/bash
#SBATCH --job-name=fix_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

RUN_DIR="runs/actinf_5k_e3"

python scripts/eval_baselines.py \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  > "${RUN_DIR}/eval_baselines.log" 2>&1

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/best_val_kl.pt" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  > "${RUN_DIR}/eval_best_val_kl.log" 2>&1

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/best_val_nss.pt" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  > "${RUN_DIR}/eval_best_val_nss.log" 2>&1

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/final.pt" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  > "${RUN_DIR}/eval_final.log" 2>&1

python scripts/plot_training_log.py \
  --log-path "${RUN_DIR}/log.csv"
