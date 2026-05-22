#!/bin/bash
#SBATCH --job-name=train_200k_sig2_res
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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

# 200k images, sigma=2.0 — resume from epoch 5 checkpoint, train to epoch 15
# scheduler rebuilt with 15-epoch total so LR continues decaying sensibly
python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/train_200k_sigma2_cosine \
  --resume runs/train_200k_sigma2_cosine/checkpoints/final.pt \
  --max-images 200000 \
  --batch-size 128 \
  --num-epochs 15 \
  --num-workers 16 \
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
  --dropout 0.1 \
  --decoder-hidden-channels 256 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --encoder-lr-scale 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --warmup-fraction 0.05 \
  --amp \
  --log-every 200 \
  --val-batches 400 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name train_200k_sigma2_cosine_resume

echo "end: $(date)"
echo "================================================================================"
