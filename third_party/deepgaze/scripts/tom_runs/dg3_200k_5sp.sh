#!/bin/bash
#SBATCH --job-name=dg3_200k_5sp
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet&80gb_vram
#SBATCH --mem=128G

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

# 200k images x 5 scanpaths: same fixations/epoch as 50k x 10sp and 100k x 5sp
# (~15M fixations/epoch, 58k batches). Extends the image-diversity scaling axis
# (1k -> 10k -> 50k -> 100k -> 200k) to show DG3's performance ceiling.
# Auto-resumes from last.pth if run directory already exists.
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
    --wandb_dir           /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs

echo "Job finished at $(date)" | tee -a "${LOG_DIR}/info.txt"
