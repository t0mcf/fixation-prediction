#!/bin/bash
#SBATCH --job-name=dg3_salicon_pretrain
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=inet

# SALICON spatial pretraining -> the checkpoint the SALICON reference arm of
# the fixed-split MIT1003 comparison starts from. The previously trained
# checkpoint is not readable from this account, so it is retrained here with
# the same recipe (scanpath/configs/salicon.yaml). 48 h walltime matches the
# recipe's historical completed run; no own throughput measurement exists yet.

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"

OUT="runs/salicon_pretrain/tom_v2"
echo "start=$(date) out=$OUT"
python -m scanpath.train_deepgaze \
    --config scanpath/configs/salicon.yaml \
    --run_name salicon_pretrain_tom_v2 \
    --out_dir "$OUT" \
    --use_wandb false
echo "end=$(date)"
