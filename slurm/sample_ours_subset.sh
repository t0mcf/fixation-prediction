#!/bin/bash
#SBATCH --job-name=ours_subscore
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-6

# Our model on the shared realism harness: sample on the frozen 400-image
# subset (5 paths/image, length 16, T=1.0), score with the same scorer and
# 16-path reference as the other models. Crop geometry (ladder convention).

#
#   sbatch slurm/sample_ours_subset.sh                                   # val subset, final.pt, T=1
#   CKPT=epoch_15.pt  TEMP=0.85  SUBSET=v2_scoring_subset_test (TEST DAY ONLY)
# CKPT selects the checkpoint file under runs/v2_<S>_5p_ll_seed42/checkpoints
# (late-checkpoint trajectory: the interim estimator wants several); TEMP is
# the pre-declared temperature-sensitivity axis (appendix only, never
# selection). Labels carry ckpt and T so rung_scores.csv stays unambiguous.

set -eo pipefail
: "${SUBSET:=v2_scoring_subset}"
: "${CKPT:=final.pt}"
: "${TEMP:=1.0}"
: "${SEED:=42}"
: "${SAMPLE_SEED:=0}"   # sampling RNG; label suffix _s<k> when != 0
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OURS_CROP_TRANSFORM=1

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
SUB=$SD/data/eval/$SUBSET
test -f "$SUB/subset_gt_5ref.json" || { echo "subset spec missing: $SUB (build it with make_v2_scoring_subset.py)" >&2; exit 4; }
SIZES=(1k 10k 50k 100k 200k 400k 800k)
S=${SIZES[$SLURM_ARRAY_TASK_ID]}
CKPT_PATH="runs/v2_${S}_5p_ll_seed${SEED}/checkpoints/$CKPT"
test -f "$CKPT_PATH" || { echo "missing: $CKPT_PATH" >&2; exit 4; }
TLAB=$(echo "$TEMP" | sed 's/\.0$//; s/\./p/')          # 1.0 -> T1, 0.85 -> T0p85
LABEL="ours_v2_${S}_${CKPT%.pt}_T${TLAB}"
[ "$SEED" != "42" ] && LABEL="${LABEL}_seed${SEED}"
[ "$SAMPLE_SEED" != "0" ] && LABEL="${LABEL}_s${SAMPLE_SEED}"
OUT=$SUB/gen/$LABEL
mkdir -p "$OUT"

python scripts/sample_for_eval.py \
  --checkpoint "$CKPT_PATH" \
  --spec "$SUB/subset_gt_5ref.json" \
  --stimuli-dir "$SUB/stimuli" --ext .JPEG \
  --out "$OUT/gen.npy" \
  --temperature "$TEMP" --seed "$SAMPLE_SEED"

PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd "$SD"
"$PY" tools/score_scanpaths.py \
  --human "$SUB/subset_gt.json" \
  --generated "$OUT/gen.npy" \
  --label "$LABEL" \
  --csv "$SUB/rung_scores.csv"
echo "done $LABEL -> $SUB/rung_scores.csv"
