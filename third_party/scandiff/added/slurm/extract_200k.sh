#!/bin/bash
#SBATCH --job-name=feat_200k
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/feat_200k_%j.out

set -e

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
FEAT_DIR=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features/fp16
IMAGES_DIR=/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"

echo "=== Extract DINOv2 features (fp16) for 200k ==="
$PY tools/precompute_features.py \
    --stems-file ./data/oursynth_200k/image_stems.txt \
    --images-dir "$IMAGES_DIR" \
    --out-dir "$FEAT_DIR" \
    --batch-size 64 \
    --fp16

echo ""
echo "=== Symlink features into dataset dirs ==="
for d in oursynth_10k_v2 oursynth_200k; do
    ln -sfn "$FEAT_DIR" "$SD/data/$d/dinov2_base_timm_image_features"
    echo "  $d -> $FEAT_DIR"
done
echo "Done."
