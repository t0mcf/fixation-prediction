#!/bin/bash
#SBATCH --job-name=dg3_mit1003_zs
#SBATCH --nodes=1
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --mem=64G

source ~/.bashrc
conda activate praktikum

export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export WANDB_ENTITY=fischer-tom-university-of-goettingen

REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
LOG_DIR="${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"

echo "Job started at $(date)" | tee "${LOG_DIR}/info.txt"
cd "${REPO_ROOT}"

# Zero-shot evaluation of DG3 imagenet-pretrained checkpoints on MIT1003.
# Models were trained ONLY on synthetic ImageNet data — no MIT1003 fine-tuning.
# Gives NSS/AUC/LL to compare against our model's zero-shot MIT1003 performance.
python -m scanpath.eval_zeroshot_mit1003 \
    --checkpoints \
        runs/imagenet_paths/dg3_1k_10sp/scanpath/best.pth \
        runs/imagenet_paths/dg3_10k_10sp/scanpath/best.pth \
        runs/imagenet_paths/dg3_50k_10sp/scanpath/best.pth \
        runs/imagenet_paths/dg3_100k_5sp/scanpath/best.pth \
    --names "DG3-1k" "DG3-10k" "DG3-50k" "DG3-100k" \
    --dataset-dir /mnt/vast-nhr/projects/nim00018/datasets \
    --num-workers 4 \
    --batch-size 32

echo "Job finished at $(date)" | tee -a "${LOG_DIR}/info.txt"
