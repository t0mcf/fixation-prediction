#!/bin/bash
#SBATCH --job-name=fix_small
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet

set -eo pipefail

echo "job id: $SLURM_JOB_ID"
echo "node: $(hostname)"
echo "start: $(date)"
echo "working directory: $(pwd)"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-none}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'

echo "python: $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -m src.training.train \
  --output-dir runs/small_1k \
  --max-images 1000 \
  --batch-size 16 \
  --max-steps 1000 \
  --num-workers 4 \
  --heatmap-sigma 2.0 \
  --heatmap-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --log-every 20 \
  --val-every 200 \
  --val-batches 20 \
  --save-every 500

echo "end: $(date)"