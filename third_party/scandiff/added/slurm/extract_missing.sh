#!/bin/bash
#SBATCH --job-name=feat_miss
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_miss_%j.out

set -e

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features/fp16
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

$PY tools/precompute_features.py \
    --stems-file ./data/missing_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR" \
    --batch-size 64 \
    --fp16
