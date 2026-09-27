#!/bin/bash
#SBATCH --job-name=feat_10k
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_10k_%j.out

set -e

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

echo "=== Step 1: Create 10k dataset JSON (canonical split) ==="
$PY tools/make_scandiff_dataset.py \
    --out-dir ./data/oursynth_10k_v2 \
    --max-images 10000 \
    --max-paths-per-image 5

echo ""
echo "=== Step 2: Extract DINOv2 features (fp16) ==="
$PY tools/precompute_features.py \
    --stems-file ./data/oursynth_10k_v2/image_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR/fp16" \
    --batch-size 64 \
    --fp16

echo ""
echo "=== Step 3: Symlink features into dataset dir ==="
ln -sfn "$FEAT_DIR/fp16" ./data/oursynth_10k_v2/dinov2_base_timm_image_features
echo "Done. Features at: $FEAT_DIR/fp16"
ls -lh ./data/oursynth_10k_v2/
