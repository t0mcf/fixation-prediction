#!/bin/bash
#SBATCH --job-name=sd_bench_1k
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -eo pipefail

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python

echo "Benchmark: 2 epochs of oursynth_1k with memmap features"
echo "start: $(date)"

$PY src/train.py \
  hydra.run.dir=./runs_ours/bench_1k/train \
  data/train_datasets=[oursynth_1k] \
  data/val_datasets=[oursynth_1k] \
  data/test_datasets=[oursynth_1k] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-bench \
  logger.wandb.name=bench_1k \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 \
  trainer.max_epochs=2 trainer.validation_every_n_epochs=1000 \
  train=true test=false

echo "end: $(date)"
