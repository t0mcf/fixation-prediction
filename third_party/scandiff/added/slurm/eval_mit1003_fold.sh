#!/bin/bash
#SBATCH --job-name=sd_eval_fold
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
# Generalizes eval_mit1003_t90.sh (fold-0-only) to any fold/mode, for the
# fold 1-3 extension of the model-vs-metric-dependence question
# (results_catalog.md §8, engineering_log.md §19.4b/c). Computes ScanDiff's
# OWN MultiMatch + ScanMatch on its own generated scanpaths (their metric
# code, not score_scanpaths.py -- that script is only for scoring OUR/DG3
# scanpaths with ScanDiff's metric code for cross-model comparison).
#   sbatch --export=ALL,FOLD=1,MODE=scratch slurm/eval_mit1003_fold.sh
#   sbatch --export=ALL,FOLD=1,MODE=ft slurm/eval_mit1003_fold.sh
set -e
: "${FOLD:?set FOLD (1,2,3,...)}"
: "${MODE:?set MODE (scratch|ft)}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python

if [ "$MODE" = "scratch" ]; then
  RUN=./runs_ours/train_mit1003_fold${FOLD}
else
  RUN=./runs_ours/ft_fold${FOLD}_av100k
fi

$PY src/eval.py \
  hydra.run.dir=$RUN/eval_fold${FOLD} \
  ckpt_path=$RUN/train/checkpoints/last.pt \
  data/train_datasets=[mit1003_fold${FOLD}] \
  data/val_datasets=[mit1003_fold${FOLD}] \
  data/test_datasets=[mit1003_fold${FOLD}] \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
  evaluation.eval_root_path=$RUN/eval_heldout \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EVAL_FOLD${FOLD}_${MODE}_DONE"
