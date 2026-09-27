#!/bin/bash
#SBATCH --job-name=dg3_fs_fair
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A_%a.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --mem=64G
#SBATCH --array=0-23

# Fair-harness evaluation of the DeepGaze III fixed-split fine-tunes
# (runs/dg3_mit1003_fixedsplit_<COND>_rep<REP>) on the authors' MIT1003
# split. These are the reported DeepGaze III results; the released weights
# were trained on all of MIT1003, including the test split.
# Same evaluator/CB reference/grids as the ours side.
#
#   sbatch scripts/tom_runs/score_dg3_fixedsplit_fair.sh                        # validation
#   sbatch --export=ALL,EVAL_SPLIT=test scripts/tom_runs/score_dg3_fixedsplit_fair.sh   # final evaluation only
# A missing run dir (fine-tune not landed) is skipped, not an error.

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${EVAL_SPLIT:=validation}"
case "$EVAL_SPLIT" in validation) OUTDIR=runs/fair_fixedsplit ;; test) OUTDIR=runs/fair_fixedsplit_test ;;
  *) echo "bad EVAL_SPLIT=$EVAL_SPLIT" >&2; exit 2 ;; esac
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
SPLITS="/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003"
TWOSIZE="$REPO_ROOT/runs/dg3_mit1003_ourfolds_scratch/MIT1003_twosize"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID}" "$REPO_ROOT/$OUTDIR"
cd "$REPO_ROOT"

CONDS=(scratch salicon 1k 10k 50k 100k 200k full1k full10k full50k full100k full200k)
COND=${CONDS[$((SLURM_ARRAY_TASK_ID / 2))]}
REP=$((SLURM_ARRAY_TASK_ID % 2))
RUN="runs/dg3_mit1003_fixedsplit_${COND}_rep${REP}"
test -d "$RUN" || { echo "skip: $RUN not present (fine-tune not landed)"; exit 0; }

python -m scanpath.eval_mit1003_fair_dg3 \
    --run-dir "$RUN" --twosize-dir "$TWOSIZE" \
    --fixed-split-dir "$SPLITS" --eval-split "$EVAL_SPLIT" \
    --name "dg3-fs-${COND}-rep${REP}" \
    2>&1 | tee "$OUTDIR/dg3_fs_${COND}_rep${REP}.log"
echo "done $COND rep$REP split=$EVAL_SPLIT $(date)"
