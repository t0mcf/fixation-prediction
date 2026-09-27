#!/bin/bash
#SBATCH --job-name=ours_frozen_test
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --constraint=inet
#SBATCH --array=0-15
#SBATCH --output=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/ours_frozen_test_%A_%a.out
#SBATCH --error=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/ours_frozen_test_%A_%a.err
set -euo pipefail
cd /user/tomcosmo.fischer/u27846/repos/fixation-prediction
set +u  # Conda's MKL activation hook reads optional unset variables.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate praktikum
set -u
export HTTPS_PROXY=http://www-cache.gwdg.de:3128
export HTTP_PROXY=http://www-cache.gwdg.de:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
python -B scripts/check_ours_validation_20260908.py
python -B scripts/run_frozen_transfer.py \
 --manifest docs/report_tables/transfer_test_manifest_ours_20260908.json \
 --outputs runs/frozen_transfer_ours_20260908 --index "$SLURM_ARRAY_TASK_ID" --split test
