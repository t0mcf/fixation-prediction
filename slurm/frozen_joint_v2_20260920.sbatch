#!/bin/bash
#SBATCH --job-name=joint_v2_frozen
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --constraint=inet
#SBATCH --output=logs/joint_v2_frozen_%A_%a.out
#SBATCH --error=logs/joint_v2_frozen_%A_%a.err
set -eo pipefail
cd "$HOME/repos/fixation-prediction"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate praktikum
export HTTPS_PROXY=http://www-cache.gwdg.de:3128
export HTTP_PROXY=http://www-cache.gwdg.de:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
echo '2bb1fc77d543c35914bf12a3c493cf7b080ea639168dc4aa65ef6f10a7e21225  docs/report_tables/transfer_test_manifest_joint_v2_20260920.json' | sha256sum -c -
python -B scripts/run_frozen_transfer.py --manifest docs/report_tables/transfer_test_manifest_joint_v2_20260920.json --outputs runs/frozen_transfer_extra_20260910 --index "$SLURM_ARRAY_TASK_ID" --split "$EVAL_SPLIT"
