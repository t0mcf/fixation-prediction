#!/bin/bash
#SBATCH --job-name=dg3_v2_sample
#SBATCH --output=logs/job_%A/output_%A_%a.txt
#SBATCH --error=logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-4
# Sample + score the synthetic-trained DG3 ladder (best.pth = best val LL) on a
# protocol-v2 scoring subset, so DG3 gets its Fig-2 curve next to ours/ScanDiff.
#   sbatch scripts/tom_runs/sample_dg3_v2_subset.sh                       # validation subset
#   sbatch --export=ALL,SUBSET=v2_scoring_subset_test scripts/tom_runs/sample_dg3_v2_subset.sh   # final evaluation only
# Walltime: unmeasured for this sampler on A100 (validation run is the measurement); 3 h cap.
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
: "${SUBSET:=v2_scoring_subset}"
: "${TEMP:=1.0}"; : "${SAMPLE_SEED:=0}"
: "${REP:=0}"   # 0 = original run dg3_<S>_5sp; k>0 = repetition dg3_<S>_5sp_rep<k>, label dg3_v2_<S>_rep<k>_best_T1 (2026-09-19)
RSUF=""; RLAB=""; [ "$REP" != "0" ] && { RSUF="_rep$REP"; RLAB="_rep$REP"; }
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
SUB=$SD/data/eval/$SUBSET
SIZES=(1k 10k 50k 100k 200k); S=${SIZES[$SLURM_ARRAY_TASK_ID]}
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}
test -f "$SUB/subset_gt_5ref.json" || { echo "missing $SUB/subset_gt_5ref.json" >&2; exit 4; }
CKPT=runs/imagenet_paths/dg3_${S}_5sp${RSUF}/scanpath/best.pth; test -f "$CKPT" || { echo "missing $CKPT" >&2; exit 4; }
TLAB=$(echo "$TEMP" | sed 's/\.0$//; s/\./p/')
LABEL="dg3_v2_${S}${RLAB}_best_T${TLAB}"; [ "$SAMPLE_SEED" != "0" ] && LABEL="${LABEL}_s${SAMPLE_SEED}"; OUT=$SUB/gen/$LABEL; mkdir -p "$OUT"
echo "JobID=$SLURM_JOB_ID SUBSET=$SUBSET SCALE=$S CKPT=$CKPT start=$(date)"
python -m scanpath.sample_scanpaths_dg3_v2 --checkpoint "$CKPT" \
  --spec "$SUB/subset_gt_5ref.json" --stimuli-dir "$SUB/stimuli" --ext .JPEG \
  --out "$OUT/gen.npy" --temperature "$TEMP" --seed "$SAMPLE_SEED"
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd "$SD"
"$PY" tools/score_scanpaths.py --human "$SUB/subset_gt.json" --generated "$OUT/gen.npy" \
  --label "$LABEL" --csv "$SUB/rung_scores.csv"
echo "done $LABEL -> $SUB/rung_scores.csv end=$(date)"
