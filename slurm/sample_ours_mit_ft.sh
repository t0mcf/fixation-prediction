#!/bin/bash
#SBATCH --job-name=ours_mit_xfam
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-15
# Cross-family whole-scanpath comparison on MIT1003: sample the FINE-TUNED
# ours checkpoints (best_val_ll, the one rule) length-matched to the human
# scanpaths (WITH initial centre fixation, as in ScanDiff's evaluator) of one fixed split and score MM/SM-KLD against them with
# ScanDiff's metric code, the same axis ScanDiff's fine-tunes are on.
# Image transform = Resize((224,224)) (sample_scanpaths.load_image without
# OURS_CROP_TRANSFORM), identical to MIT fine-tuning/evaluation.
#   sbatch slurm/sample_ours_mit_ft.sh                        # validation split
#   sbatch --export=ALL,SPLIT=test slurm/sample_ours_mit_ft.sh  # final evaluation only
set -eo pipefail
: "${SPLIT:=validation}"; : "${TEMP:=1.0}"; : "${SAMPLE_SEED:=0}"
: "${TRAIN_SEEDS:=42:43}"
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh; conda activate praktikum
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OURS_CROP_TRANSFORM
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
SPEC=$SD/data/eval/mit1003_human_initial_${SPLIT}.json; test -f "$SPEC" || { echo "missing $SPEC (tools/extract_mit1003_human_initial.py)" >&2; exit 4; }
CONDS=(scratch bv1k bv10k bv50k bv100k bv200k bv400k bv800k)
IFS=':' read -r -a SEEDS <<< "$TRAIN_SEEDS"
NSEEDS=${#SEEDS[@]}
C=${CONDS[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % NSEEDS))]}
CKPT=runs/ft_authsplit_${C}_seed${SEED}/checkpoints/best_val_ll.pt; test -f "$CKPT" || { echo "missing $CKPT" >&2; exit 4; }
TLAB=$(echo "$TEMP" | sed 's/\.0$//; s/\./p/'); LABEL="ours_ft_${C}_seed${SEED}_T${TLAB}"; [ "$SAMPLE_SEED" != "0" ] && LABEL="${LABEL}_s${SAMPLE_SEED}"
OUT=$SD/data/eval/mit_xfam_${SPLIT}/gen/$LABEL; mkdir -p "$OUT"
echo "JobID=$SLURM_JOB_ID SPLIT=$SPLIT COND=$C SEED=$SEED CKPT=$CKPT start=$(date)"
python scripts/sample_for_eval.py --checkpoint "$CKPT" --spec "$SPEC" \
  --stimuli-dir /mnt/vast-nhr/projects/nim00018/datasets/MIT1003/stimuli --ext .jpeg \
  --out "$OUT/gen.npy" --temperature "$TEMP" --seed "$SAMPLE_SEED"
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd "$SD"; "$PY" tools/score_scanpaths.py --human "$SPEC" --generated "$OUT/gen.npy" \
  --label "$LABEL" --csv "$SD/data/eval/mit_xfam_${SPLIT}/scores.csv"
echo "done $LABEL end=$(date)"
