#!/bin/bash
#SBATCH --job-name=sd_ft_pilot
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet

# PIPELINE PILOT, NOT a citable result: validates that init_from (weights-only
# fine-tune init, src/trainer.py) works end-to-end -- data loads, loss goes
# down, MultiMatch/ScanMatch eval runs -- before committing GPU time to the
# full 10-fold pretrain-vs-pretrain comparison.
#
# Uses the existing mit1003_t90 split (= our seed-42 fold-0) and a FROZEN copy
# of the train_100k ladder checkpoint at epoch 118 (runs_ours/pilot_checkpoints/
# train_100k_epoch118.pt), copied out before the ladder's own resume job could
# advance it further -- so this run's init point is exactly known and stable
# even while the ladder keeps training in parallel.
set -e
: "${EPOCHS:=30}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/ft_pilot_av100k_fold0
INIT=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/pilot_checkpoints/train_100k_epoch118.pt
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[mit1003_t90] \
  data/val_datasets=[mit1003_t90] \
  data/test_datasets=[mit1003_t90] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=ft_pilot_av100k_fold0 \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=5 \
  init_from=${INIT} \
  train=true test=false
echo "FT_PILOT_DONE"
