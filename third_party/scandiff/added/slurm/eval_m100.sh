#!/bin/bash
#SBATCH --job-name=sd_eval_m100
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet

# Generate ScanDiff scanpaths on the mit1003_100 eval set for one trained
# ladder checkpoint. Usage: sbatch --export=ALL,SIZE=50k slurm/eval_m100.sh
# (SIZE names runs_ours/train_${SIZE} and the oursynth_${SIZE} dataset config.)
set -e
: "${SIZE:?Set SIZE (e.g. 1k, 10k_v2, 50k, 100k, 200k)}"
: "${CKPT:=last.pt}"          # e.g. samples6M.pt for the samples-matched point
: "${EVALTAG:=eval_mit100}"   # output subdir; use eval_mit100_6M for snapshots
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_${SIZE}
$PY src/eval.py \
  hydra.run.dir=$RUN/${EVALTAG}_hydra \
  ckpt_path=$RUN/train/checkpoints/${CKPT} \
  data/train_datasets=[oursynth_${SIZE}] \
  data/val_datasets=[mit1003_100] \
  data/test_datasets=[mit1003_100] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
  evaluation.eval_root_path=$RUN/${EVALTAG} \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EVAL_M100 DONE SIZE=${SIZE}"
