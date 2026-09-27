#!/bin/bash
#SBATCH --job-name=sd_gen_smoke
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --constraint=inet
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
# 4 images (shard 0 of 2500), reduced steps — format smoke only
$PY scripts/generate_pretrain_scandiff.py \
  --dataset oursynth_10k --sp-per-image 16 --respacing 100 \
  --shard 0 --num-shards 2500 \
  --out /tmp/scandiffgen_smoke.parquet
cp /tmp/scandiffgen_smoke.parquet data/scandiffgen_10k/smoke.parquet
echo "GEN SMOKE DONE"
