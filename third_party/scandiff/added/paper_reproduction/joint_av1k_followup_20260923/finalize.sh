#!/bin/bash
#SBATCH --job-name=sd_joint1k_finalize
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/joint_av1k_followup_20260923/finalize_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/joint_av1k_followup_20260923/finalize_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=inet
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python -B /mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/joint_av1k_followup_20260923/finalize.py
