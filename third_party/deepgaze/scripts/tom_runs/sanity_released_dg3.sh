#!/bin/bash
#SBATCH --job-name=dg3_released_sanity
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=inet

# Two reference evaluations of the authors' RELEASED DeepGaze III weights on
# the fixed validation split (contaminated upper-bound reference — released
# weights saw all of MIT1003):
#   1. through OUR fair harness (shared grids + tuned CB reference)
#   2. through pysaliency's own log-likelihood machinery (native resolution)
# Agreement between the two validates the fair harness against the reference
# implementation for the one model both pipelines score natively.

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   EVAL_SPLIT=test (final evaluation only) -> runs/fair_fixedsplit_test/
: "${EVAL_SPLIT:=validation}"
case "$EVAL_SPLIT" in validation) OUTDIR=runs/fair_fixedsplit ;; test) OUTDIR=runs/fair_fixedsplit_test ;;
  *) echo "bad EVAL_SPLIT=$EVAL_SPLIT" >&2; exit 2 ;; esac
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
SPLITS="/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003"
TWOSIZE="$REPO_ROOT/runs/dg3_mit1003_ourfolds_scratch/MIT1003_twosize"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID}" "$REPO_ROOT/$OUTDIR"
cd "$REPO_ROOT"

python -m scanpath.eval_mit1003_fair_dg3 \
    --released --twosize-dir "$TWOSIZE" \
    --fixed-split-dir "$SPLITS" --eval-split "$EVAL_SPLIT" \
    --name dg3-released-fair \
    2>&1 | tee "$OUTDIR/dg3_released_fair.log"

python -m scanpath.pysaliency_sanity_dg3 \
    --released --twosize-dir "$TWOSIZE" \
    --fixed-split-dir "$SPLITS" \
    --name dg3-released-pysaliency \
    2>&1 | tee "$OUTDIR/dg3_released_pysaliency.log"

echo "done $(date)"
