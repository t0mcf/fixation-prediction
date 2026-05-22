#!/bin/bash
#SBATCH --job-name=profile_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --constraint=inet

set -eo pipefail

mkdir -p logs

echo "================================================================================"
echo "JobID = ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "start: $(date)"
echo "================================================================================"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python scripts/profile_model.py \
  --batch-size 128 \
  --seq-len 16 \
  --fusion-layers 4 \
  --paths-per-image 16 \
  --loss ll \
  --amp \
  --warmup-steps 3 \
  --profile-steps 5 \
  --output-dir runs/profile \
  --export-trace

echo "end: $(date)"
echo "================================================================================"
