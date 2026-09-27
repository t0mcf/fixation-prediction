#!/bin/bash
#SBATCH --job-name=sd_train_avcrop
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail

ROOT=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_10k_v2_avcrop
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd "$ROOT"

test ! -e "$RUN/train/checkpoints/last.pt" || {
  echo "Refusing to silently restart an existing run: $RUN" >&2
  exit 1
}

$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[oursynth_10k_v2_avcrop] \
  data/val_datasets=[oursynth_10k_v2_avcrop] \
  data/test_datasets=[oursynth_10k_v2_avcrop] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=oursynth_10k_v2_avcrop \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=12 data.pin_memory=true data.batch_size_validation=128 \
  trainer.max_epochs=120 trainer.validation_every_n_epochs=5 \
  train=true test=false tags=[oursynth_10k_v2_avcrop]
