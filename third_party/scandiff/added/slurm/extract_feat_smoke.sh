#!/bin/bash
#SBATCH --job-name=scandiff_feat_smoke
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
$PY tools/extract_dinov2_features.py \
  --images-list data/oursynth_smoke/images_list.txt \
  --out-dir data/oursynth_smoke/dinov2_base_timm_image_features \
  --device cuda
