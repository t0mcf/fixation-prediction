#!/bin/bash
#SBATCH --job-name=fix_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -m src.training.train \
  --output-dir runs/actinf_5k_e3 \
  --max-images 5000 \
  --num-epochs 3 \
  --batch-size 16 \
  --num-workers 4 \
  --paths-per-image 16 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-fraction 0.05 \
  --log-every 50 \
  --val-batches 50 \
  --amp \
  --use-wandb \
  --wandb-mode online \
  --wandb-project fixation-prediction \
  --wandb-entity tomcosmo-fischer \
  --wandb-name actinf_5k_e3

echo "end: $(date)"
