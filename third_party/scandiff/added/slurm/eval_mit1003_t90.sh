#!/bin/bash
#SBATCH --job-name=sd_eval_t90
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet

# Validation anchor: generate with the REAL-MIT1003-trained ScanDiff (t90)
# on its 101 held-out images; score externally (their ScanMatch crashes).
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_mit1003_t90
$PY src/eval.py \
  hydra.run.dir=$RUN/eval_t90 \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  data/train_datasets=[mit1003_t90] \
  data/val_datasets=[mit1003_t90] \
  data/test_datasets=[mit1003_t90] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
  evaluation.eval_root_path=$RUN/eval_heldout \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EVAL_T90 DONE"
