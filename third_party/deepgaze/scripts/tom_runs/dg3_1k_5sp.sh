#!/bin/bash
#SBATCH --job-name=dg3_1k_5sp
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-24:00:00
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
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}"
echo "start $(date)"
cd "${REPO_ROOT}"

# provenance snapshot
RUN_DIR="${REPO_ROOT}/runs/imagenet_paths/dg3_1k_5sp"
mkdir -p "${RUN_DIR}/code_snapshot"
cp -r scanpath "${RUN_DIR}/code_snapshot/" 2>/dev/null || true
git rev-parse HEAD > "${RUN_DIR}/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "${RUN_DIR}/code_snapshot/workdir.diff" 2>/dev/null || true

# DG3 fill-in for the fixed-5sp v2 scaling curve: the 1k point existed only at
# 10 scanpaths/image (path-count mismatch with every other rung and with
# ours/ScanDiff). Same seed/val protocol as the rest of the ladder — the
# selection is the verified v2 train_1k prefix (verified 2026-08-24).
python -m scanpath.train_deepgaze \
    --config scanpath/configs/imagenet.yaml \
    --wandb_project fixation-prediction --run_name dg3_1k_5sp \
    --batch_size 256 --num_workers 16 --in_memory false \
    --train_n_images 1000 --train_n_scanpaths 5 \
    --val_n_images 5000 --val_n_scanpaths 10 \
    --lr 0.002 --use_wandb true \
    --wandb_dir /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs
echo "end $(date)"
