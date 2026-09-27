#!/bin/bash
#SBATCH --job-name=fair_ours
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --constraint=inet
#SBATCH --array=0-37

# Fair-harness evaluation of every ft_authsplit checkpoint on the fixed
# split: shared grids (64/128/224), tuned train-only CB reference, the same
# accumulator that scored DeepGaze III, so the cross-model table reads in one
# convention. min_fixations=1 everywhere since 2026-09-02.
#
#   sbatch slurm/fair_fixedsplit_ours.sh                      # validation
#   EVAL_SPLIT=test sbatch --export=ALL,EVAL_SPLIT=test ...    # final evaluation only
#
# Conditions cover the final-init ladder, the best-val-init ladder (primary
# since 2026-09-02), the triangle arms and the shuffled placebo; a missing
# checkpoint (arm not trained yet) is skipped, not an error.

set -eo pipefail
: "${EVAL_SPLIT:=validation}"
case "$EVAL_SPLIT" in validation) OUTDIR=runs/fair_fixedsplit ;; test) OUTDIR=runs/fair_fixedsplit_test ;;
  *) echo "bad EVAL_SPLIT=$EVAL_SPLIT" >&2; exit 2 ;; esac
mkdir -p logs "$OUTDIR"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDS=(scratch lad1k lad10k lad50k lad100k lad200k lad400k lad800k
       bv1k bv10k bv50k bv100k bv200k bv400k bv800k
       sd5 avfull avmatch shuffled)
COND=${CONDS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=$((42 + SLURM_ARRAY_TASK_ID % 2))
CKPT="runs/ft_authsplit_${COND}_seed${SEED}/checkpoints/best_val_ll.pt"
test -f "$CKPT" || { echo "skip: $CKPT not present (arm not trained)"; exit 0; }
OUT="$OUTDIR/ours_${COND}_seed${SEED}.log"

python scripts/eval_mit1003_fair.py \
  --mode finetuned \
  --checkpoint "$CKPT" \
  --mit-split-dir /mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003 \
  --eval-split "$EVAL_SPLIT" \
  --name "ours-${COND}-seed${SEED}" \
  2>&1 | tee "$OUT"
echo "done $COND seed$SEED split=$EVAL_SPLIT"
