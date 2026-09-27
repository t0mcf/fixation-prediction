#!/bin/bash
#SBATCH --job-name=dg3_fair_smoke
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --mem=64G

source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}

# SMOKE: pipeline check of the fair-harness DG3 wrapper on the existing
# fold-0 smoke checkpoint (trained on pysaliency's split — numbers are
# NOT valid, leakage w.r.t. our folds; this only validates the code path).
python -m scanpath.eval_mit1003_fair_dg3 \
    --run-dir runs/dg3_mit1003_from100k_smoke \
    --ckpt-name best.pth \
    --name dg3-smoke-INVALID-NUMBERS \
    --limit-folds 1 --limit-batches 40 \
    --resolutions 64 128 224
echo SMOKEDONE
