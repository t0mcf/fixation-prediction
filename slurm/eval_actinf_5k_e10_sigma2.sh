#!/bin/bash
#SBATCH --job-name=eval_5k_e10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -eo pipefail

mkdir -p logs

echo "================================================================================"
echo "JobID = ${SLURM_JOB_ID}"
echo "User = ${USER}"
echo "Partition = ${SLURM_JOB_PARTITION}"
echo "Nodelist = ${SLURM_JOB_NODELIST}"
echo "================================================================================"
echo "job id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "start: $(date)"
echo "working directory: $(pwd)"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-none}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

RUN_DIR="runs/actinf_5k_e10_sigma2_noamp"
PARQUET="/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet"
IMAGENET="/mnt/vast-nhr/projects/nim00018/datasets/ImageNet"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

echo ""
echo "================================================================================"
echo "Evaluating baselines"
echo "================================================================================"

python scripts/eval_baselines.py \
  --parquet-path "${PARQUET}" \
  --imagenet-root "${IMAGENET}" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --max-prefix-len 15 \
  --center-sigma 12.0 \
  --density-smoothing 1.0 \
  --eval-batches 50 \
  | tee "${RUN_DIR}/baselines_eval.log"

echo ""
echo "================================================================================"
echo "Evaluating best_val_kl checkpoint"
echo "================================================================================"

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/best_val_kl.pt" \
  --parquet-path "${PARQUET}" \
  --imagenet-root "${IMAGENET}" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  | tee "${RUN_DIR}/model_best_val_kl_eval.log"

echo ""
echo "================================================================================"
echo "Evaluating best_val_nss checkpoint"
echo "================================================================================"

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/best_val_nss.pt" \
  --parquet-path "${PARQUET}" \
  --imagenet-root "${IMAGENET}" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  | tee "${RUN_DIR}/model_best_val_nss_eval.log"

echo ""
echo "================================================================================"
echo "Evaluating final checkpoint"
echo "================================================================================"

python scripts/evaluate_checkpoint.py \
  --checkpoint "${RUN_DIR}/checkpoints/final.pt" \
  --parquet-path "${PARQUET}" \
  --imagenet-root "${IMAGENET}" \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --eval-batches 50 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  | tee "${RUN_DIR}/model_final_eval.log"

echo ""
echo "================================================================================"
echo "Done"
echo "end: $(date)"
echo "================================================================================"
