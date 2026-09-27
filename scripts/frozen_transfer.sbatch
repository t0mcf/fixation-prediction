#!/bin/bash
#SBATCH --job-name=frozen_transfer
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --constraint=inet
#SBATCH --array=0-29
#SBATCH --output=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/frozen_transfer_%A_%a.out
#SBATCH --error=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/frozen_transfer_%A_%a.err
set -euo pipefail
: "${MANIFEST:?absolute path to audited manifest}"
: "${OUTPUTS:?absolute path to new result directory}"
: "${EVAL_SPLIT:=validation}"
cd /user/tomcosmo.fischer/u27846/repos/fixation-prediction
set +u  # Conda's MKL activation hook reads optional unset variables.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate praktikum
set -u
export HTTPS_PROXY=http://www-cache.gwdg.de:3128
export HTTP_PROXY=http://www-cache.gwdg.de:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
python -B scripts/run_frozen_transfer.py --manifest "$MANIFEST" --outputs "$OUTPUTS" \
    --index "$SLURM_ARRAY_TASK_ID" --split "$EVAL_SPLIT"
