#!/bin/bash
#SBATCH --job-name=sd_train
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch --export=SIZE=10k_v2,EPOCHS=200 slurm/train_scaling.sh

set -e

: "${SIZE:?Set SIZE (e.g. 1k, 10k_v2, 50k, 100k, 200k)}"
: "${EPOCHS:=200}"

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_${SIZE}

echo "Training ScanDiff on oursynth_${SIZE} for ${EPOCHS} epochs"

$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[oursynth_${SIZE}] \
  data/val_datasets=[oursynth_${SIZE}] \
  data/test_datasets=[oursynth_${SIZE}] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=oursynth_${SIZE} \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=5 \
  train=true test=false \
  tags=[oursynth_${SIZE}]

echo "Done: $RUN"
