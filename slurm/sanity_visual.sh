#!/bin/bash
#SBATCH --job-name=sanity_visual
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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

# 8 images, overfit setup — just enough to see whether predictions are smooth.
# log images every 10 steps so we get visuals early in wandb.
python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/sanity_visual \
  --max-images 8 \
  --batch-size 16 \
  --num-epochs 100 \
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
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --warmup-fraction 0.0 \
  --log-every 10 \
  --image-log-every 10 \
  --val-batches 10 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name sanity_visual_nearest_conv

echo "end: $(date)"
echo "================================================================================"
