#!/bin/bash
#SBATCH --job-name=fullval_ourmetrics
#SBATCH --output=logs/job_%A/output_%A_%a.txt
#SBATCH --error=logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-3

# Re-score the DG3 in-domain ladder with fixation-prediction's metric
# implementations on a common 64x64 grid, so DG3 and our model are measured by
# the same code. DG3's own nss() is not NSS (std/mean swap, see the module
# docstring of scanpath/eval_imagenet_fullval.py); its LL and AUC are fine and
# are reported alongside as a cross-check.
#
# Only the four rungs trained with five scanpaths per image are included; the
# 1k rung uses ten and is not protocol-matched to our ladder.
#
# The July run of the same evaluation took roughly 20-25 min per checkpoint,
# so two hours per array task is comfortable.

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export TORCH_HOME="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
cd "$REPO_ROOT"
mkdir -p "logs/job_${SLURM_JOB_ID}" runs/fullval_eval_ourmetrics

NAMES=(dg3_10k_5sp dg3_50k_5sp dg3_100k_5sp dg3_200k_5sp)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=runs/imagenet_paths/${NAME}/scanpath/best.pth
CONFIG=runs/imagenet_paths/${NAME}/scanpath/config.json
JSON=runs/fullval_eval_ourmetrics/${NAME}.json
OUT=runs/fullval_eval_ourmetrics/${NAME}.log

echo "JobID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=$(hostname) start=$(date)" | tee "$OUT"
echo "name=${NAME} checkpoint=${CKPT}" | tee -a "$OUT"

python -m scanpath.eval_imagenet_fullval \
  --checkpoint "$CKPT" \
  --config-json "$CONFIG" \
  --name "$NAME" \
  --batch-size 128 \
  --num-workers 8 \
  --our-metrics \
  --our-grid 64 \
  --out-json "$JSON" | tee -a "$OUT"

echo "end=$(date)" | tee -a "$OUT"
