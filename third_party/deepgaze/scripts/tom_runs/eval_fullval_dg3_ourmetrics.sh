#!/bin/bash
#SBATCH --job-name=fullval_dg3
#SBATCH --output=logs/job_%A/output_%A_%a.txt
#SBATCH --error=logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-4

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export TORCH_HOME="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
cd "$REPO_ROOT"
# VAL_IMAGE_LIST=.../test_10k.txt (final evaluation only) evaluates the protocol test
# split instead of the seed-3141 canonical validation set (== validation_5k).
: "${VAL_IMAGE_LIST:=}"
OUTDIR=runs/fullval_eval_ourmetrics
LIST_ARG=""
if [ -n "$VAL_IMAGE_LIST" ]; then OUTDIR=runs/fullval_eval_ourmetrics_test; LIST_ARG="--val-image-list $VAL_IMAGE_LIST"; fi
mkdir -p "logs/job_${SLURM_JOB_ID}" "$OUTDIR"

NAMES=(
  dg3_1k_5sp
  dg3_10k_5sp
  dg3_50k_5sp
  dg3_100k_5sp
  dg3_200k_5sp
)
CKPTS=(
  runs/imagenet_paths/dg3_1k_5sp/scanpath/best.pth
  runs/imagenet_paths/dg3_10k_5sp/scanpath/best.pth
  runs/imagenet_paths/dg3_50k_5sp/scanpath/best.pth
  runs/imagenet_paths/dg3_100k_5sp/scanpath/best.pth
  runs/imagenet_paths/dg3_200k_5sp/scanpath/best.pth
)
CONFIGS=(
  runs/imagenet_paths/dg3_1k_5sp/scanpath/config.json
  runs/imagenet_paths/dg3_10k_5sp/scanpath/config.json
  runs/imagenet_paths/dg3_50k_5sp/scanpath/config.json
  runs/imagenet_paths/dg3_100k_5sp/scanpath/config.json
  runs/imagenet_paths/dg3_200k_5sp/scanpath/config.json
)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
OUT=$OUTDIR/${NAME}.log
JSON=$OUTDIR/${NAME}.json

echo "================================================================================" | tee "$OUT"
echo "JobID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=$(hostname) start=$(date)" | tee -a "$OUT"
echo "name=${NAME}" | tee -a "$OUT"
echo "checkpoint=${CKPT}" | tee -a "$OUT"
echo "protocol=${VAL_IMAGE_LIST:-full canonical val: 5000 images}, all 16 scanpaths/image" | tee -a "$OUT"
echo "================================================================================" | tee -a "$OUT"

python -m scanpath.eval_imagenet_fullval \
  --checkpoint "$CKPT" \
  --config-json "$CONFIG" \
  --name "$NAME" \
  --batch-size 128 \
  --num-workers 8 $LIST_ARG \
  --our-metrics --out-json "$JSON" | tee -a "$OUT"

echo "end=$(date)" | tee -a "$OUT"
