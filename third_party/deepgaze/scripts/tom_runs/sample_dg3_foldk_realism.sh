#!/bin/bash
#SBATCH --job-name=dg3_sample_foldk
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
# Generalized version of sample_dg3_fold0_realism.sh -- any CV fold, not just 0.
#   sbatch --export=ALL,FOLD=1 scripts/tom_runs/sample_dg3_foldk_realism.sh
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
: "${FOLD:?set FOLD}"
mkdir -p logs/job_${SLURM_JOB_ID:-local}
SPEC=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval/mit1003_human_fold${FOLD}.json
STIM=/mnt/vast-nhr/projects/nim00018/datasets/MIT1003/stimuli
OUT=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval
B=/mnt/vast-nhr/projects/nim00018/tom/scanpather-artifacts

echo "=== sampling DG3-scratch (fold${FOLD}) ==="
python -m scanpath.sample_scanpaths_dg3 \
  --checkpoint "$B/runs/dg3_mit1003_ourfolds_scratch/MIT1003_scanpath_partially_frozen_saliency_network/crossval-10-${FOLD}/best.pth" \
  --spec "$SPEC" --stimuli-dir "$STIM" \
  --out "$OUT/gen_fold${FOLD}_dg3-scratch.npy" \
  --temperature 1.0

echo "=== sampling DG3-100k-finetuned (fold${FOLD}) ==="
python -m scanpath.sample_scanpaths_dg3 \
  --checkpoint "$B/runs/dg3_mit1003_ourfolds_100k/MIT1003_scanpath_partially_frozen_saliency_network/crossval-10-${FOLD}/best.pth" \
  --spec "$SPEC" --stimuli-dir "$STIM" \
  --out "$OUT/gen_fold${FOLD}_dg3-100k.npy" \
  --temperature 1.0

echo "SAMPLING DONE FOLD${FOLD}"
