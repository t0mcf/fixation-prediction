#!/bin/bash
#SBATCH --job-name=frozen_scandiff_test
#SBATCH --output=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/%x_%A_%a.out
#SBATCH --error=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-13%2
set -euo pipefail
: "${MANIFEST:?required}"
: "${MANIFEST_SHA256:?required}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
exec /mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python -B \
 /user/tomcosmo.fischer/u27846/repos/fixation-prediction/scripts/frozen_scandiff_transfer.py \
 --manifest "$MANIFEST" --sha256 "$MANIFEST_SHA256" --action test --index "$SLURM_ARRAY_TASK_ID"
