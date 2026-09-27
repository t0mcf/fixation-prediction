#!/bin/bash
#SBATCH --job-name=synth_fair
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet

# One run of scripts/eval_synthetic_fair.py (shared in-domain evaluator,
# ours or DG3, explicit image list, frozen-runner metrics.json schema).
# Validation-side use only unless SYNTHETIC_TEST_OK=1 is exported deliberately.
#
#   sbatch --export=ALL,MODEL=ours,CKPT=runs/v2_1k_5p_ll_seed42/checkpoints/best_val_ll.pt,\
#          NAME=ours_1k_seed42,OUT=runs/synthetic_fair/validation/ours_1k_seed42/metrics.json \
#          slurm/eval_synthetic_fair.sh
#   sbatch --export=ALL,MODEL=dg3,CKPT=<scanpather>/runs/imagenet_paths/dg3_1k_5sp/scanpath/best.pth,\
#          CONFIG=<...>/scanpath/config.json,NAME=dg3_1k,OUT=... slurm/eval_synthetic_fair.sh
#   LIST defaults to validation_5k.txt. EXTRA passes further flags (e.g. --amp).
set -eo pipefail
: "${MODEL:?ours|dg3}"; : "${CKPT:?}"; : "${NAME:?}"; : "${OUT:?}"
: "${LIST:=/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av/validation_5k.txt}"
REPO=${REPO:-$HOME/repos/fixation-prediction}
cd "$REPO"; mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export TORCH_HOME="$HOME/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG_ARG=""; [ -n "$CONFIG" ] && CFG_ARG="--config-json $CONFIG"
echo "JobID=${SLURM_JOB_ID} node=$(hostname) start=$(date) MODEL=$MODEL NAME=$NAME LIST=$LIST"
python scripts/eval_synthetic_fair.py --model "$MODEL" --checkpoint "$CKPT" $CFG_ARG \
  --image-list "$LIST" --name "$NAME" --out-json "$OUT" \
  --batch-size "${BS:-64}" --num-workers 8 ${EXTRA:-}
echo "end=$(date)"
