#!/bin/bash
#SBATCH --job-name=scandiff_eval_10k_m100
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_10k
$PY src/eval.py \
  hydra.run.dir=$RUN/eval_m100 \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  data/train_datasets=[oursynth_10k] \
  data/val_datasets=[mit1003_100] \
  data/test_datasets=[mit1003_100] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
  evaluation.eval_root_path=$RUN/eval_mit100 \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EVAL_M100 DONE"
