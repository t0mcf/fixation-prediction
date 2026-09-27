#!/bin/bash
#SBATCH --job-name=dg3_100k_5sp_resume
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-29:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet&80gb_vram
#SBATCH --mem=128G

# Continuation of dg3_100k_5sp stage 2 (found 2026-09-02): the
# original 48 h job spent ~18 h on stage 1 and reached only epoch 8 of the
# 10-epoch stage-2 budget (LR still 2e-4, val LL still rising). Stage 1 is
# complete (spatial/final.pth, 30/30) and is reused; _train() restores
# scanpath/last.pth (model+optimizer+scheduler, step 8) and runs epochs 9-10.
# Walltime: measured 8 h 33 min per stage-2 epoch (log.csv timestamps,
# epoch 7->8) x 2 epochs = 17.1 h, x1.7 buffer = 29 h.
# Invocation otherwise identical to scripts/tom_runs/dg3_100k_10sp.sh (which,
# despite its file name, is the 5sp launcher: run_name dg3_100k_5sp).

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

echo "Job started at $(date)"         | tee    "${LOG_DIR}/info.txt"
echo "Node:  $(hostname)"             | tee -a "${LOG_DIR}/info.txt"
echo "GPUs:  ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_DIR}/info.txt"

cd "${REPO_ROOT}"
test -f runs/imagenet_paths/dg3_100k_5sp/scanpath/last.pth || { echo "no stage-2 last.pth to resume from" >&2; exit 4; }
test -f runs/imagenet_paths/dg3_100k_5sp/spatial/final.pth  || { echo "stage-1 final.pth missing" >&2; exit 4; }

python -m scanpath.train_deepgaze \
    --config              scanpath/configs/imagenet.yaml \
    --wandb_project       fixation-prediction \
    --run_name            dg3_100k_5sp \
    --batch_size          256 \
    --num_workers         16 \
    --in_memory           false \
    --train_n_images      100000 \
    --train_n_scanpaths   5 \
    --val_n_images        5000 \
    --val_n_scanpaths     10 \
    --lr                  0.002 \
    --use_wandb           true \
    --wandb_dir           /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs \
    --stage1.path         runs/imagenet_paths/dg3_100k_5sp/spatial \
    --stage1.use_pretrained true

echo "Job finished at $(date)" | tee -a "${LOG_DIR}/info.txt"
