#!/bin/bash
#SBATCH --job-name=sd_gen10k
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=inet

# Phase 2: t90 teacher generates 16 scanpaths/image on the oursynth_10k
# ImageNet images, full 1000 diffusion steps, sharded 8 ways (~1250 img/shard
# ≈ 5-6h at the measured ~240 img/h). Output: our-format parquet shards.
#   sbatch --array=0-7 slurm/gen_pretrain_10k.sh
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
S=${SLURM_ARRAY_TASK_ID:?array required}
$PY scripts/generate_pretrain_scandiff.py \
  --dataset oursynth_10k \
  --sp-per-image 16 \
  --respacing 1000 \
  --shard $S --num-shards 8 \
  --out data/scandiffgen_10k/shard_${S}.parquet
echo "GEN SHARD $S DONE"
