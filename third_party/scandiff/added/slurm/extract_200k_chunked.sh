#!/bin/bash
#SBATCH --job-name=feat_chunk
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_chunk_%j.out

set -e
export PYTHONUNBUFFERED=1

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features/chunked
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

mkdir -p "$FEAT_DIR"

$PY tools/precompute_features.py \
    --stems-file ./data/oursynth_200k/image_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR" \
    --batch-size 64 \
    --chunk-size 1000 \
    --fp16

echo "=== Done ==="
