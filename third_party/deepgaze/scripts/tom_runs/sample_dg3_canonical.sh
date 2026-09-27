#!/bin/bash
#SBATCH --job-name=sample_cv_dg3
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A_%a.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet
# DeepGaze III scanpaths on the SYNTHETIC canonical val set (in-domain scaling).
#   sbatch --array=0-2 scripts/tom_runs/sample_dg3_canonical.sh
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
SCALES=(10k 50k 200k)
S=${SCALES[${SLURM_ARRAY_TASK_ID:-0}]}
CV=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval/canonical_val
echo "=== dg3-${S} on canonical val ==="
python -m scanpath.sample_scanpaths_dg3 \
  --checkpoint runs/imagenet_paths/dg3_${S}_5sp/scanpath/best.pth \
  --spec $CV/canonical_val_gt.json \
  --stimuli-dir $CV/stimuli --ext .JPEG \
  --out $CV/gen_cv_dg3_${S}.npy \
  --temperature 1.0
echo "DONE dg3 ${S}"
