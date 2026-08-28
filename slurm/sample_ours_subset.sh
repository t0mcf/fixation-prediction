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

set -eo pipefail
mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OURS_CROP_TRANSFORM=1

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
SUB=$SD/data/eval/v2_scoring_subset
SIZES=(1k 10k 50k 100k 200k 400k 800k)
S=${SIZES[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/v2_${S}_5p_ll_seed42/checkpoints/final.pt"
test -f "$CKPT" || { echo "missing: $CKPT" >&2; exit 4; }
OUT=$SUB/gen/ours_v2_${S}_final_T1
mkdir -p "$OUT"

python scripts/sample_for_eval.py \
  --checkpoint "$CKPT" \
  --spec "$SUB/subset_gt_5ref.json" \
  --stimuli-dir "$SUB/stimuli" --ext .JPEG \
  --out "$OUT/gen.npy" \
  --temperature 1.0 --seed 0

PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd "$SD"
"$PY" tools/score_scanpaths.py \
  --human "$SUB/subset_gt.json" \
  --generated "$OUT/gen.npy" \
  --label "ours_v2_${S}_final_T1" \
  --csv "$SUB/rung_scores.csv"
echo "done ours_v2_${S}"
