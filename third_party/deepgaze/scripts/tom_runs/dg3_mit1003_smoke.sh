#!/bin/bash
#SBATCH --job-name=dg3_mit_smoke
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-04:00:00
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

# SMOKE TEST: DG3 MIT1003 fine-tuning, SINGLE fold (fold 0 of 10), from our
# DG3 ImageNet 100k checkpoint (scanpath_pretrain_path => skips SALICON spatial stage).
# Verifies the finicky pysaliency MIT1003 pipeline runs end-to-end before committing 10 folds.
python -m scanpath.train_deepgaze \
    --config scanpath/configs/mit1003.yaml \
    --scanpath_pretrain_path runs/imagenet_paths/dg3_100k_5sp/scanpath/best.pth \
    --crossval_folds 10 \
    --fold_only 0 \
    --out_dir runs/dg3_mit1003_from100k_smoke \
    --run_name dg3_mit1003_from100k_smoke \
    --use_wandb true \
    --wandb_dir /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs
echo "end $(date)"
