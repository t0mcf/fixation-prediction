#!/bin/bash
#SBATCH --job-name=dg3_fair_score
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --mem=64G

# Score all three split-aligned DG3 MIT1003 CV conditions through the shared
# fair harness (same folds, same tuned CB reference, same evaluator as ours).
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}

for c in scratch 100k 200k; do
  echo "############ DG3 ${c} (our folds, fair harness) ############"
  python -m scanpath.eval_mit1003_fair_dg3 \
    --run-dir runs/dg3_mit1003_ourfolds_${c} \
    --stage MIT1003_scanpath \
    --name dg3-${c}-ourfolds \
    --resolutions 64 128 224
done
echo ALLDONE
