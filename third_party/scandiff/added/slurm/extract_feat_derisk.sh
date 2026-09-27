#!/bin/bash
#SBATCH --job-name=scandiff_feat_derisk
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
echo "=== MIT1003 features (fp16) ==="
$PY tools/extract_dinov2_features.py --images-list data/mit1003/images_list.txt \
  --out-dir data/mit1003/dinov2_base_timm_image_features --fp16 --batch-size 32 --device cuda
echo "=== 10k synthetic features (fp16) ==="
$PY tools/extract_dinov2_features.py --images-list data/oursynth_10k/images_list.txt \
  --out-dir data/oursynth_10k/dinov2_base_timm_image_features --fp16 --batch-size 32 --device cuda
echo "ALL DERISK FEATURES DONE"
