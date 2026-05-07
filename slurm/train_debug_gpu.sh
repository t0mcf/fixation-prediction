#!/bin/bash
#SBATCH --job-name=fix_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --constraint=inet

set -eo pipefail

echo "job id: $SLURM_JOB_ID"
echo "node: $(hostname)"
echo "start: $(date)"
echo "working directory: $(pwd)"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-none}"

source ~/miniforge3/etc/profile.d/conda.sh
set +u
conda activate praktikum
set -u

export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'

echo "python: $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python scripts/train_debug.py \
  --parquet-path data/debug_scanpaths.parquet \
  --max-images 5 \
  --batch-size 8 \
  --max-steps 100 \
  --num-workers 2 \
  --heatmap-sigma 2.0 \
  --heatmap-size 64 \
  --lr 1e-4 \
  --output-dir runs/debug_gpu \
  --save-checkpoint

python scripts/eval_debug.py \
  --checkpoint runs/debug_gpu/checkpoint_debug.pt \
  --parquet-path data/debug_scanpaths.parquet \
  --max-images 5 \
  --batch-size 8 \
  --num-workers 2 \
  --max-batches 10 \
  --heatmap-sigma 2.0 \
  --heatmap-size 64

echo "end: $(date)"