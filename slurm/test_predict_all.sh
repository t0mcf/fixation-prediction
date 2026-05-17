#!/bin/bash
#SBATCH --job-name=test_predict_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# overfit test: 8 images, batch_size=16 -> 8 steps/epoch, 500 epochs = 4000 steps
# predict-all gives 15x more gradient signal per step vs random-t
python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/predict_all_test \
  --max-images 8 \
  --batch-size 16 \
  --num-epochs 500 \
  --num-workers 4 \
  --seed 42 \
  --paths-per-image 16 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --max-seq-len 16 \
  --visual-encoder dinov2_base \
  --d-model 256 \
  --n-heads 8 \
  --scanpath-layers 2 \
  --fusion-layers 4 \
  --dropout 0.0 \
  --decoder-hidden-channels 256 \
  --lr 1e-4 \
  --lr-schedule constant \
  --encoder-lr-scale 0.1 \
  --weight-decay 0.00 \
  --grad-clip 1.0 \
  --warmup-fraction 0.0 \
  --log-every 50 \
  --val-batches 50 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name predict_all_test

echo "end: $(date)"
echo "================================================================================"
