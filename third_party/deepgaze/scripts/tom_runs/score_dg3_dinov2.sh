#!/bin/bash
#SBATCH --job-name=dg3_dino_score
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --mem=64G
# Score the DG3+DINOv2 backbone-swap ablation through the shared fair harness.
# Missing folds are skipped automatically, so this runs on whatever is trained.
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}
python -m scanpath.eval_mit1003_fair_dg3 \
  --run-dir runs/dg3_mit1003_ourfolds_dinov2scratch \
  --stage MIT1003_scanpath \
  --backbone dinov2 \
  --name dg3-dinov2scratch-ourfolds \
  --resolutions 64
echo ALLDONE
