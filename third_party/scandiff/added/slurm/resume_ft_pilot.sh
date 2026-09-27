#!/bin/bash
#SBATCH --job-name=sd_ft_pilot_resume
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet

# The 30-epoch pilot (ft_pilot_av100k_fold0) scored MM-KLD 0.160, much worse
# than train_mit1003_t90's 200-epoch from-scratch 0.052 -- but that is an
# epoch-budget confound, not evidence pretraining hurts: t90 trained 200
# epochs, the pilot only 30 (deliberately, as a pipeline smoke test). This
# resumes the SAME fine-tune run (full ckpt_path resume: optimizer + epoch
# counter preserved, NOT init_from) up to 200 epochs total, matching t90's
# budget, so the two are comparable before drawing any conclusion.
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/ft_pilot_av100k_fold0
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[mit1003_t90] \
  data/val_datasets=[mit1003_t90] \
  data/test_datasets=[mit1003_t90] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=ft_pilot_av100k_fold0_resume \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=4 data.batch_size_validation=128 \
  trainer.max_epochs=200 trainer.validation_every_n_epochs=5 \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  train=true test=false
echo "FT_PILOT_RESUME_DONE"
