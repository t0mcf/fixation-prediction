#!/bin/bash
#SBATCH --job-name=eval_v2_ladder
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-13

# In-domain density metrics (NSS/LL/AUC/KL) for the full v2 ladder, both
# selection rules, on the complete 5k canonical validation set.
# --eval-batches -1 disables the batch cap; the crop geometry must be set
# explicitly because it is env-driven, not stored in the checkpoint.

#
#   sbatch slurm/eval_v2_ladder_indomain.sh                                  # validation_5k
#   sbatch --export=ALL,IMAGE_LIST=.../test_10k.txt slurm/eval_v2_ladder_indomain.sh   # TEST DAY ONLY
#   SEED=43 for the replicate ladder (runs/v2_*_5p_ll_seed43).

set -eo pipefail
: "${IMAGE_LIST:=}"
: "${SEED:=42}"
OUTDIR=runs/v2_eval_indomain
LIST_ARG=""
if [ -n "$IMAGE_LIST" ]; then
  case "$IMAGE_LIST" in *test_10k*) OUTDIR=runs/v2_eval_indomain_test ;; *) OUTDIR=runs/v2_eval_indomain_$(basename "${IMAGE_LIST%.txt}") ;; esac
  LIST_ARG="--image-list $IMAGE_LIST"
fi
[ "$SEED" != "42" ] && OUTDIR="${OUTDIR}_seed${SEED}"
mkdir -p logs "$OUTDIR"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# v2 protocol geometry — the ladder was TRAINED with the crop transform
export OURS_CROP_TRANSFORM=1

SIZES=(1k 10k 50k 100k 200k 400k 800k)
S=${SIZES[$((SLURM_ARRAY_TASK_ID / 2))]}
CKPTS=(final.pt best_val_ll.pt)
C=${CKPTS[$((SLURM_ARRAY_TASK_ID % 2))]}
RUN="runs/v2_${S}_5p_ll_seed${SEED}"
CKPT="$RUN/checkpoints/$C"
test -f "$CKPT" || { echo "missing: $CKPT" >&2; exit 4; }
OUT="$OUTDIR/${S}_${C%.pt}.log"

echo "SIZE=$S SEED=$SEED CKPT=$C IMAGE_LIST=${IMAGE_LIST:-validation_5k(canonical)} OURS_CROP_TRANSFORM=$OURS_CROP_TRANSFORM"
python scripts/evaluate_checkpoint.py \
  --checkpoint "$CKPT" \
  --eval-batches -1 $LIST_ARG \
  --batch-size 64 --num-workers 8 \
  2>&1 | tee "$OUT"
echo "done $S $C -> $OUT"
