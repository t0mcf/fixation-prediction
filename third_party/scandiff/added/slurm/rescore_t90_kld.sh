#!/bin/bash
#SBATCH --job-name=sd_rescore_kld
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --constraint=inet
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python tools/rescore_t90_kld.py
echo "RESCORE_DONE"
