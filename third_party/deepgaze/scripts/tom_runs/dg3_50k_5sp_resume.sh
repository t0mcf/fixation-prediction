#!/bin/bash
#SBATCH --job-name=dg3_50k_5sp_resume
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-23:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet&80gb_vram
#SBATCH --mem=128G

# Continuation of dg3_50k_5sp stage 2 (found 2026-09-02): the
# original 24 h job reached only epoch 6 of the 10-epoch stage-2 budget
# (LR 2e-4, second milestone at 8 never reached). Stage 1 is complete
# (spatial/final.pth, 30/30) and is reused; _train() restores
# scanpath/last.pth (model+optimizer+scheduler, step 6) and runs epochs 7-10.
# Walltime: measured 3 h 19 min per stage-2 epoch (log.csv timestamps,
# epochs 4->5->6) x 4 epochs = 13.3 h, x1.7 buffer = 23 h.
# Invocation otherwise identical to scripts/tom_runs/dg3_50k_5sp.sh.

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
test -f runs/imagenet_paths/dg3_50k_5sp/scanpath/last.pth || { echo "no stage-2 last.pth to resume from" >&2; exit 4; }
test -f runs/imagenet_paths/dg3_50k_5sp/spatial/final.pth  || { echo "stage-1 final.pth missing" >&2; exit 4; }

python -m scanpath.train_deepgaze \
    --config scanpath/configs/imagenet.yaml \
    --wandb_project fixation-prediction --run_name dg3_50k_5sp \
    --batch_size 256 --num_workers 16 --in_memory false \
    --train_n_images 50000 --train_n_scanpaths 5 \
    --val_n_images 5000 --val_n_scanpaths 10 \
    --lr 0.002 --use_wandb true \
    --wandb_dir /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/wandb_runs \
    --stage1.path runs/imagenet_paths/dg3_50k_5sp/spatial \
    --stage1.use_pretrained true
echo "end $(date)"
