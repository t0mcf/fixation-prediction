#!/bin/bash
#SBATCH --job-name=fixed_seq_eval
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet
set -euo pipefail
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python tools/run_fixed_sequence_eval.py
