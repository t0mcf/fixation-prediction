#!/bin/bash
#SBATCH --job-name=restripe
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=standard96:shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -eo pipefail
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
bash tools/restripe_memmap.sh "${NAME:?set NAME}"
