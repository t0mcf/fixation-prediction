#!/bin/bash
#SBATCH --job-name=ours_joint_human
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet
#SBATCH --array=0-3
#SBATCH --output=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/ours_joint_%A_%a.out
#SBATCH --error=/user/tomcosmo.fischer/u27846/repos/fixation-prediction/logs/ours_joint_%A_%a.err
set -euo pipefail
REPO=/user/tomcosmo.fischer/u27846/repos/fixation-prediction
cd "$REPO"
set +u  # Conda's MKL activation hook reads optional unset variables.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate praktikum
set -u
export OURS_CROP_TRANSFORM=0
export PYTHONDONTWRITEBYTECODE=1
export HTTPS_PROXY=http://www-cache.gwdg.de:3128
export HTTP_PROXY=http://www-cache.gwdg.de:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
CONDS=(scratch bv800k scratch bv800k)
SEEDS=(42 42 43 43)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
: "${RUNSUF:=_v2}"   # _v2 = COCO start-fixation fix of 2026-09-19; the unsuffixed runs are superseded
RUN="$REPO/runs/ft_joint_authsplit_${COND}_seed${SEED}${RUNSUF}"
test ! -e "$RUN" || { echo "refusing existing run: $RUN"; exit 3; }
INIT=()
if [ "$COND" = bv800k ]; then
  INIT=(--init-from "$REPO/runs/v2_800k_5p_ll_seed42/checkpoints/best_val_ll.pt")
fi
mkdir -p "$RUN/code_snapshot/scripts"
cp -r src "$RUN/code_snapshot/"
cp scripts/train_joint_mit_coco.py "$RUN/code_snapshot/scripts/"
cp slurm/ft_joint_mit_coco.sbatch "$RUN/code_snapshot/"
cd "$RUN/code_snapshot"
python -B scripts/train_joint_mit_coco.py \
 --dataset mit1003 --mit-split-dir /mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003 \
 --output-dir "$RUN" --batch-size 32 --num-epochs 30 --num-workers 8 --seed "$SEED" \
 --max-seq-len 16 --heatmap-size 64 --heatmap-sigma 2.0 \
 --visual-encoder dinov2_base --d-model 256 --n-heads 8 --scanpath-layers 2 --fusion-layers 4 \
 --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
 --lr 1e-4 --lr-schedule cosine --weight-decay 0.01 --grad-clip 1.0 --warmup-fraction 0.05 \
 --loss ll --amp --log-every 50 --val-batches -1 "${INIT[@]}"
