#!/bin/bash
#SBATCH --job-name=sd_feat_avcrop
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
set -euo pipefail

ROOT=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd "$ROOT"

$PY tools/extract_dinov2_features_avcrop.py \
  --images-list data/oursynth_10k_v2/image_stems.txt \
  --images-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images \
  --out-dir data/oursynth_10k_v2/dinov2_base_timm_image_features_avcrop \
  --av-center-crop --fp16 --batch-size 32 --device cuda
$PY tools/extract_dinov2_features_avcrop.py \
  --images-list runs_ours/canonical_val/images_list.txt \
  --out-dir runs_ours/canonical_val/features_avcrop \
  --av-center-crop --fp16 --batch-size 32 --device cuda

$PY - <<'PY'
from pathlib import Path
import torch

checks = [
    (Path("data/oursynth_10k_v2/dinov2_base_timm_image_features_avcrop"), 15000),
    (Path("runs_ours/canonical_val/features_avcrop"), 400),
]
for directory, expected in checks:
    files = sorted(directory.glob("*.pth"))
    assert len(files) == expected, (directory, len(files), expected)
    for path in (files[0], files[len(files) // 2], files[-1]):
        value = torch.load(path, map_location="cpu")
        assert tuple(value.shape) == (1369, 768), (path, value.shape)
        assert torch.isfinite(value).all(), path
    print(f"validated {len(files)} finite features in {directory}")
PY
