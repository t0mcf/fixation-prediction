#!/bin/bash
#SBATCH --job-name=sd_spatial
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# Does a thesis-v2 ladder checkpoint show the spatial displacement that broke
# the 2026-08-20 MIT1003 retrain? Samples a small slice of each rung's own
# validation split and reports where the fixations land.
#   sbatch --array=0-2 slurm/check_v2_spatial.sh
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
SIZES=(1k 10k 50k)
S=${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}
/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python \
  scripts/check_v2_spatial.py --size "$S" --respacing 1000 --limit 0.02
