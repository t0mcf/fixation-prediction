#!/bin/bash
#SBATCH --job-name=bench_feat
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/benchmark_%j.out

set -e

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface

cd "$SD"
$PY tools/benchmark_features.py --data-dir ./data/oursynth_smoke --images-dir /mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images --n-batches 50 --batch-size 32 --num-workers 4
