#!/bin/bash
#SBATCH --job-name=sd_score_avcrop
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet
set -euo pipefail

ROOT=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd "$ROOT"

test -f runs_ours/train_10k_avcrop/train/checkpoints/last.pt
test "$(find runs_ours/canonical_val/features_avcrop -maxdepth 1 -name '*.pth' | wc -l)" -eq 400
$PY scripts/gen_and_score_canonical_variant.py --size 10k --variant _avcrop --respacing 1000
