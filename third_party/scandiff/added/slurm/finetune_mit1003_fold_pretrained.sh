#!/bin/bash
#SBATCH --job-name=sd_ft_fold
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet
# av100k-pretrained -> MIT1003 fold-K fine-tune (generalizes
# finetune_mit1003_pilot.sh / resume_ft_pilot.sh, which were fold-0 only).
# Weights-only init (init_from), full 200-epoch budget from the start (the
# fold-0 pilot needed a resume because it started at a 30-epoch smoke-test
# budget; this goes straight to 200 to avoid that detour).
#   sbatch --export=ALL,FOLD=1 slurm/finetune_mit1003_fold_pretrained.sh
set -e
: "${EPOCHS:=200}"
: "${FOLD:?set FOLD (1,2,3,...)}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/ft_fold${FOLD}_av100k
INIT=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/pilot_checkpoints/train_100k_epoch118.pt
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[mit1003_fold${FOLD}] \
  data/val_datasets=[mit1003_fold${FOLD}] \
  data/test_datasets=[mit1003_fold${FOLD}] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=ft_fold${FOLD}_av100k \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=10 \
  init_from=${INIT} \
  train=true test=false
echo "FT_FOLD${FOLD}_PRETRAINED_DONE"
