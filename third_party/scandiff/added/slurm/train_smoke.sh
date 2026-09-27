#!/bin/bash
#SBATCH --job-name=scandiff_train_smoke
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
$PY src/train.py \
  data/train_datasets=[oursynth] \
  data/val_datasets=[oursynth] \
  data/test_datasets=[oursynth] \
  trainer=gpu \
  callbacks=default \
  diffusion.num_timesteps=1000 \
  diffusion_class=spaced_diffusion \
  data.num_workers=8 \
  trainer.max_epochs=2 \
  trainer.validation_every_n_epochs=1000 \
  train=true test=false \
  tags=[scandiff_smoke]
