#!/bin/bash
#SBATCH --job-name=scandiff_train_eval_10k
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_10k

echo "############## TRAIN (oursynth_10k, 10 epochs) ##############"
$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[oursynth_10k] \
  data/val_datasets=[oursynth_10k] \
  data/test_datasets=[mit1003] \
  trainer=gpu callbacks=default \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=16 \
  trainer.max_epochs=10 trainer.validation_every_n_epochs=1000 \
  train=true test=false \
  tags=[oursynth_10k]

echo "############## EVAL on MIT1003 (MultiMatch + ScanMatch no-dur) ##############"
$PY src/eval.py \
  hydra.run.dir=$RUN/eval \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  data/train_datasets=[oursynth_10k] \
  data/test_datasets=[mit1003] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
  evaluation.eval_root_path=$RUN/eval_mit1003 \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "############## DERISK TRAIN+EVAL DONE ##############"
