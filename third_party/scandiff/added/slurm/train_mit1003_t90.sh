#!/bin/bash
#SBATCH --job-name=sd_mit1003_t90
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=inet

# Phase 1 of ScanDiff-as-data-generator: train ScanDiff on REAL MIT1003
# (902 train images, held-out 101 = our seed-42 fold-0 val -> leak-free
# teacher for transfer evals). Doubles as the validation-anchor run
# (compare vs their published MIT1003 row to show we train them sanely).
set -e
: "${EPOCHS:=200}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_mit1003_t90
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[mit1003_t90] \
  data/val_datasets=[mit1003_t90] \
  data/test_datasets=[mit1003_t90] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=mit1003_t90 \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=10 \
  train=true test=false
echo "TRAIN_MIT1003_T90 DONE"
