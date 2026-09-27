#!/bin/bash
#SBATCH --job-name=sd_eval_ft_pilot
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet

# Score the ft_pilot_av100k_fold0 checkpoint (ScanDiff pretrained on
# active-vision 100k, then weights-only fine-tuned 30 epochs on mit1003_t90 =
# our seed-42 fold-0) on the SAME held-out 101 images as train_mit1003_t90
# (scratch-on-human teacher), so the two are directly comparable.
# 'kld' requested up front this time (yesterday's t90 eval silently skipped it).
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/ft_pilot_av100k_fold0
$PY src/eval.py \
  hydra.run.dir=$RUN/eval_heldout \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  data/train_datasets=[mit1003_t90] \
  data/val_datasets=[mit1003_t90] \
  data/test_datasets=[mit1003_t90] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur','kld'] \
  evaluation.eval_root_path=$RUN/eval_heldout \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EVAL_FT_PILOT_DONE"
