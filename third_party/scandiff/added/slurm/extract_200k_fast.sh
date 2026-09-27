#!/bin/bash
#SBATCH --job-name=feat_fast
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_fast_%j.out

set -e
export PYTHONUNBUFFERED=1

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features/fp16
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images
LOCAL=/local/jobs/${SLURM_JOB_ID}/feat_staging

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

mkdir -p "$LOCAL"
echo "Local staging dir: $LOCAL"
df -h /local

$PY tools/precompute_features.py \
    --stems-file ./data/oursynth_200k/image_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR" \
    --staging-dir "$LOCAL" \
    --flush-every 40000 \
    --batch-size 64 \
    --fp16

echo "=== Done ==="
