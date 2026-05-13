#!/bin/bash
#SBATCH --job-name=5k_e10_f4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet

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

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/actinf_5k_e10_fusion4_sigma2_noamp \
  --max-images 5000 \
  --batch-size 16 \
  --num-epochs 10 \
  --num-workers 4 \
  --seed 42 \
  --paths-per-image 16 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --visual-encoder dinov2_base \
  --d-model 256 \
  --n-heads 8 \
  --scanpath-layers 2 \
  --fusion-layers 4 \
  --dropout 0.1 \
  --decoder-hidden-channels 256 \
  --max-prefix-len 15 \
  --lr 1e-4 \
  --encoder-lr-scale 0.1 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --warmup-fraction 0.05 \
  --log-every 50 \
  --val-batches 50 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name actinf_5k_e10_fusion4_sigma2_noamp

echo "end: $(date)"
echo "================================================================================"
