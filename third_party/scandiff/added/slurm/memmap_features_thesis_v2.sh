#!/bin/bash
#SBATCH --job-name=sdv2_memmap
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=standard96:shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/data/thesis_v2
CHUNKS=/mnt/lustre-grete/workspaces/ws/nim00018/u27846-nim18-dinov2-features/chunked_thesis_v2_avcrop
OUT=/mnt/lustre-grete/workspaces/ws/nim00018/u27846-nim18-dinov2-features/memmap_thesis_v2_avcrop_200k
cd "$SD"
"$PY" tools/chunks_to_memmap_v1.py --chunks "$CHUNKS" \
  --images-list "$DATA/oursynth_v2_200k_feature_images.tsv" --out "$OUT"
for scale in 1k 10k 50k 100k 200k; do
  d="$DATA/oursynth_v2_${scale}"
  ln -sfn "$OUT" "$d/dinov2_base_timm_image_features"
done
