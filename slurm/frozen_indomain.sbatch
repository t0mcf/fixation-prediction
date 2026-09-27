#!/bin/bash
#SBATCH --job-name=frozen_indomain
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-14
set -eo pipefail
: "${MANIFEST:?absolute manifest path required}"
: "${MANIFEST_SHA256:?required}"
: "${RECEIPT:?absolute validation receipt required}"
: "${RECEIPT_SHA256:?required}"
cd /user/tomcosmo.fischer/u27846/repos/fixation-prediction
export PYTHONDONTWRITEBYTECODE=1
export HTTPS_PROXY=http://www-cache.gwdg.de:3128
export HTTP_PROXY=http://www-cache.gwdg.de:3128
export TORCH_HOME="$HOME/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$HOME/miniforge3/envs/praktikum/bin/python" -B scripts/run_frozen_indomain.py \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
  --action test --index "$SLURM_ARRAY_TASK_ID" \
  --receipt "$RECEIPT" --receipt-sha256 "$RECEIPT_SHA256"
