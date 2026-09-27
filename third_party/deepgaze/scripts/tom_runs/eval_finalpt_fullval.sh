#!/bin/bash
#SBATCH --job-name=dg3_finaleval
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A_%a.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --constraint=inet
#SBATCH --array=0-3

# final.pth full-val evals for the DG3 ladder — the fixed-endpoint counterpart
# to the existing best.pth numbers, so DG3 reports both selection rules like
# every other model (fixed final primary / best-val secondary). Same protocol
# as runs/fullval_eval/*.json: full 5k val, 16 paths/image, seed 3141.

source ~/.bashrc
conda activate praktikum
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export MALLOC_ARENA_MAX=2
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_ARRAY_JOB_ID:-local}"

RUNS=(dg3_10k_5sp dg3_50k_5sp dg3_100k_5sp dg3_200k_5sp)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
# fixed training endpoint: final.pth where it exists (our 10k fill-in),
# last.pth otherwise (the earlier runs save last every epoch; run completed
# => last == end of training). The eval json records the exact file used.
CKPT="runs/imagenet_paths/${RUN}/scanpath/final.pth"
[ -f "$CKPT" ] || CKPT="runs/imagenet_paths/${RUN}/scanpath/last.pth"
test -f "$CKPT" || { echo "missing endpoint checkpoint for $RUN" >&2; exit 4; }

python -m scanpath.eval_imagenet_fullval \
    --checkpoint "$CKPT" \
    --name "${RUN}_final" \
    --out-json "runs/fullval_eval/${RUN}_final.json"
echo "done ${RUN}_final"
