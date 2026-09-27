#!/bin/bash
#SBATCH --job-name=sdv2_features
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/data/thesis_v2
OUT=/mnt/lustre-grete/workspaces/ws/nim00018/u27846-nim18-dinov2-features/chunked_thesis_v2_avcrop
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
cd "$SD"
mkdir -p "$OUT"
"$PY" tools/precompute_features_v1.py \
  --images-list "$DATA/oursynth_v2_200k_feature_images.tsv" \
  --images-dir /mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images \
  --out-dir "$OUT" --batch-size 64 --num-workers 16 --chunk-size 1000 --fp16 --av-center-crop
