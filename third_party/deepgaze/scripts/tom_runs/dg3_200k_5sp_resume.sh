#!/bin/bash
#SBATCH --job-name=dg3_200k_5sp_resume
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet&80gb_vram
#SBATCH --mem=128G

# Continuation leg for dg3_200k_5sp stage 2 (2026-09-02): the run
# stopped at epoch 2 of 10 (LR never decayed). One stage-2 epoch measured
# ~38 h (log.csv timestamps epoch 1->2), so one leg = one epoch; chain legs
# with --dependency=afterany. Not expected to reach the 10-epoch budget
# before the compute freeze — purpose is to show whether the truncated point
# keeps rising, not to produce a budget-matched rung. Stage 1 (spatial,
# 30/30) is reused; _train() restores scanpath/last.pth incl. optimizer and
# scheduler. Submitted with --nice so it never outranks the pending
# fine-tunes and replicates.
#   sbatch --nice=2000 scripts/tom_runs/dg3_200k_5sp_resume.sh
#   sbatch --nice=2000 --dependency=afterany:<prev> scripts/tom_runs/dg3_200k_5sp_resume.sh

source ~/.bashrc
conda activate praktikum

export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export MALLOC_ARENA_MAX=2
export WANDB_ENTITY=fischer-tom-university-of-goettingen
export WANDB_DIR=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
LOG_DIR="${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"
echo "Job started at $(date)" | tee "${LOG_DIR}/info.txt"
echo "Node:  $(hostname)"     | tee -a "${LOG_DIR}/info.txt"

cd "${REPO_ROOT}"
test -f runs/imagenet_paths/dg3_200k_5sp/scanpath/last.pth || { echo "no stage-2 last.pth to resume from" >&2; exit 4; }
test -f runs/imagenet_paths/dg3_200k_5sp/spatial/final.pth  || { echo "stage-1 final.pth missing" >&2; exit 4; }

python -m scanpath.train_deepgaze \
    --config              scanpath/configs/imagenet.yaml \
    --wandb_project       fixation-prediction \
    --run_name            dg3_200k_5sp \
    --batch_size          256 \
    --num_workers         16 \
    --in_memory           false \
    --train_n_images      200000 \
    --train_n_scanpaths   5 \
    --val_n_images        5000 \
    --val_n_scanpaths     10 \
    --lr                  0.002 \
    --use_wandb           true \
    --wandb_dir           /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs \
    --stage1.path         runs/imagenet_paths/dg3_200k_5sp/spatial \
    --stage1.use_pretrained true

echo "Job finished at $(date)" | tee -a "${LOG_DIR}/info.txt"
