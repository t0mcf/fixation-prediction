#!/bin/bash
#SBATCH --job-name=sd_scratch_fold
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet
# Scratch-on-human ScanDiff training for CV fold K (generalizes
# train_mit1003_t90.sh, which is hardcoded to fold 0). Companion arm to
# finetune_mit1003_fold_pretrained.sh for the multi-fold model-vs-metric
# comparison (engineering_log.md 19.4b/c).
#   sbatch --export=ALL,FOLD=1 slurm/train_mit1003_fold_scratch.sh
set -e
: "${EPOCHS:=200}"
: "${FOLD:?set FOLD (1,2,3,...)}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_mit1003_fold${FOLD}
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[mit1003_fold${FOLD}] \
  data/val_datasets=[mit1003_fold${FOLD}] \
  data/test_datasets=[mit1003_fold${FOLD}] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=mit1003_fold${FOLD}_scratch \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=10 \
  train=true test=false
echo "TRAIN_MIT1003_FOLD${FOLD}_SCRATCH_DONE"
