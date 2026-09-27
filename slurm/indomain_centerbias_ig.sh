#!/bin/bash
#SBATCH --job-name=indomain_cb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=medium96s
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
# CPU job: scripts/scanpath_geometry.py needs ~30 GB for the two 500k-scanpath
# corpora and gets OOM-killed on the login node.
set -eo pipefail
cd $HOME/repos/fixation-prediction; mkdir -p logs
source ~/miniforge3/etc/profile.d/conda.sh; conda activate praktikum
python scripts/indomain_centerbias_ig.py --n-fit-images 20000
