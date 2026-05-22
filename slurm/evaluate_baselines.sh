#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# evaluate uniform, centre gaussian, and empirical density baselines on the val set
# run this before comparing model checkpoints to know what to beat
python -m scripts.evaluate_baselines \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --max-images 5000 \
  --batch-size 32 \
  --num-workers 8 \
  --seed 42 \
  --eval-batches 200 \
  --density-batches 500 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --center-sigma 16.0 \
  --max-seq-len 16

echo "end: $(date)"
echo "================================================================================"
