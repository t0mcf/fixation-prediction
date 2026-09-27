#!/bin/bash
#SBATCH --job-name=ours_pysaliency
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=inet

# pysaliency reference evaluation of OUR fine-tuned checkpoints on the fixed
# validation split (pixel-grid LL, the maximally DG3-favourable endpoint).
#   sbatch --export=ALL,COND=lad800k,SEED=42 scripts/tom_runs/pysaliency_ours.sh

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
FIXPRED="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/fixation-prediction"
SPLITS="/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003"
TWOSIZE="$REPO_ROOT/runs/dg3_mit1003_ourfolds_scratch/MIT1003_twosize"
COND="${COND:?e.g. scratch|lad800k}"
SEED="${SEED:-42}"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID}" "$REPO_ROOT/runs/fair_fixedsplit"
cd "$REPO_ROOT"

python -m scanpath.pysaliency_eval_ours \
    --checkpoint "$FIXPRED/runs/ft_authsplit_${COND}_seed${SEED}/checkpoints/best_val_ll.pt" \
    --fixed-split-dir "$SPLITS" --twosize-dir "$TWOSIZE" \
    --name "ours-${COND}-seed${SEED}" \
    2>&1 | tee "runs/fair_fixedsplit/ours_pysaliency_${COND}_seed${SEED}.log"
echo "done $(date)"
