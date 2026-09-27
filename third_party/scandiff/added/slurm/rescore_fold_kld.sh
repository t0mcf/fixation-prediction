#!/bin/bash
#SBATCH --job-name=sd_rescore_kld
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Rescore an already-saved fold1-3 generation with KLD added (tools/rescore_fold_kld.py).
# Needs a GPU purely because the saved .pkl generations are CUDA-tagged tensors.
#   sbatch --export=ALL,FOLD=1,MODE=scratch slurm/rescore_fold_kld.sh
set -e
: "${FOLD:?set FOLD}"
: "${MODE:?set MODE (scratch|ft)}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
$PY tools/rescore_fold_kld.py --fold "$FOLD" --mode "$MODE"
echo "RESCORE_FOLD${FOLD}_${MODE}_DONE"
