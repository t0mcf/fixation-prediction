#!/bin/bash
#SBATCH --job-name=sd_r1_score
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%A_%a.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet
#SBATCH --array=10,30,50,70,90,110,130,150,170,190,199

# Offline metric trajectory for R1 (train_r1_official_recipe.sh): score the
# every-10-epoch checkpoints through the authors' own MIT1003-test evaluation
# — the identical pipeline that produced the released-checkpoint reference
# numbers (MM KLD 0.0397 / SM 0.0411 / SS 0.0469). This replaces the authors'
# 40 in-loop sampling passes; the numbers are computed by the same code either
# way. Array runs every 20 epochs plus the final epoch; densify around the
# best region afterwards if needed (sbatch --array=<epochs> this script).
#
# epoch 199's checkpoint is last.pt (saved after the final epoch).

set -euo pipefail
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
RUN=$SD/paper_reproduction/r1_official_recipe_seed0
E=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$E" = "199" ]; then
  CKPT=$RUN/train/checkpoints/last.pt
else
  CKPT=$RUN/train/checkpoints/epoch_${E}.pt
fi
test -f "$CKPT" || { echo "checkpoint missing: $CKPT" >&2; exit 4; }

export CKPT
export TAG="r1_eval_epoch_${E}"
bash "$SD/paper_reproduction/eval_reproduced_freeview_mit1003.sh"
