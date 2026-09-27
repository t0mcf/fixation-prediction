#!/bin/bash
#SBATCH --job-name=dg3_mit_xfam
#SBATCH --output=logs/job_%A/output_%A_%a.txt
#SBATCH --error=logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-13
# Cross-family whole-scanpath comparison on MIT1003 for the FINE-TUNED DG3
# fixed-split runs (stage 3 best.pth): sample length-matched to the human
# scanpaths (with initial fixation) at native twosize resolution with the
# train-only center bias, then score MM/SM-KLD with ScanDiff's metric code.
#   sbatch scripts/tom_runs/sample_dg3_ft_mit.sh                        # validation
#   sbatch --export=ALL,SPLIT=test scripts/tom_runs/sample_dg3_ft_mit.sh  # final evaluation only
# Walltime unmeasured (3 h cap); the validation run is the measurement.
set -eo pipefail
source ~/.bashrc; conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'; export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
: "${SPLIT:=validation}"; : "${TEMP:=1.0}"; : "${SAMPLE_SEED:=0}"
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
SPEC=$SD/data/eval/mit1003_human_initial_${SPLIT}.json; test -f "$SPEC" || { echo "missing $SPEC" >&2; exit 4; }
SPLITS=/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003
CONDS=(scratch salicon full1k full10k full50k full100k full200k)
C=${CONDS[$((SLURM_ARRAY_TASK_ID / 2))]}; REP=$((SLURM_ARRAY_TASK_ID % 2))
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather; mkdir -p logs/job_${SLURM_JOB_ID:-local}
RUN=runs/dg3_mit1003_fixedsplit_${C}_rep${REP}; test -f "$RUN/MIT1003_scanpath/fixedsplit/best.pth" || { echo "missing $RUN stage-3 best.pth" >&2; exit 4; }
TLAB=$(echo "$TEMP" | sed 's/\.0$//; s/\./p/'); LABEL="dg3_ft_${C}_rep${REP}_T${TLAB}"; [ "$SAMPLE_SEED" != "0" ] && LABEL="${LABEL}_s${SAMPLE_SEED}"
OUT=$SD/data/eval/mit_xfam_${SPLIT}/gen/$LABEL; mkdir -p "$OUT"
echo "JobID=$SLURM_JOB_ID SPLIT=$SPLIT COND=$C REP=$REP start=$(date)"
python -m scanpath.sample_scanpaths_dg3_ft --run-dir "$RUN" \
  --twosize-dir runs/dg3_mit1003_ourfolds_scratch/MIT1003_twosize \
  --fixed-split-dir "$SPLITS" --eval-split "$SPLIT" --spec "$SPEC" \
  --out "$OUT/gen.npy" --temperature "$TEMP" --seed "$SAMPLE_SEED"
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
cd "$SD"; "$PY" tools/score_scanpaths.py --human "$SPEC" --generated "$OUT/gen.npy" \
  --label "$LABEL" --csv "$SD/data/eval/mit_xfam_${SPLIT}/scores.csv"
echo "done $LABEL end=$(date)"
