#!/bin/bash
#SBATCH --job-name=feat_vast
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_vast_%j.out

set -e
export PYTHONUNBUFFERED=1

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/dinov2_features_fp16
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

echo "=== Copying existing features from lustre to VAST ==="
LUSTRE=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features/fp16
rsync -a "$LUSTRE/" "$FEAT_DIR/"
echo "=== Copy done, starting extraction ==="

$PY tools/precompute_features.py \
    --stems-file ./data/oursynth_200k/image_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR" \
    --batch-size 64 \
    --fp16

echo "=== Done ==="
