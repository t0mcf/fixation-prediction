#!/bin/bash
#SBATCH --job-name=v2_800k_36ep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

# Epoch-budget control for the top rung (decided 2026-09-02): the 800k rung's
# best validation epoch is 17 of 18, i.e. at the end of its cosine schedule,
# so "converged" and "budget-limited" cannot be told apart from the ladder
# alone. This run is identical to slurm/v2_ladder.sh SIZE=800k except for
# --num-epochs 36 (2x; the cosine schedule stretches with it, so this tests
# budget sufficiency under a proportionally longer schedule, not merely more
# steps) and its own run name. It is not a ladder point and is never plotted
# on the ladder; it is reported as a control in 5.1.
#
# Walltime: 800k x 5 paths / 128 = 31,250 steps/epoch; the 18-epoch run
# needed ~2.5 h/epoch (12 h legs, ~5 epochs/leg). 36 epochs ~ 90 h ->
# three 48 h legs chained with afterany; each leg resumes from latest.pt and
# an unused link exits immediately. Submit with --nice so it never outranks
# the pending fine-tunes/replicates:
#   j1=$(sbatch --nice=2000 --parsable slurm/v2_800k_36ep_control.sh)
#   j2=$(sbatch --nice=2000 --parsable --dependency=afterany:$j1 slurm/v2_800k_36ep_control.sh)
#   j3=$(sbatch --nice=2000 --parsable --dependency=afterany:$j2 slurm/v2_800k_36ep_control.sh)
set -eo pipefail
SIZE=800k
: "${SEED:=42}"
REPO=${REPO:-$HOME/repos/fixation-prediction}
V2=/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av
RUN="runs/v2_${SIZE}_36ep_5p_ll_seed${SEED}"
CKPT="$RUN/checkpoints/latest.pt"

cd "$REPO"
mkdir -p logs
test -f "$V2/train_${SIZE}.txt"
RESUME_ARG=""
if [ -f "$CKPT" ]; then RESUME_ARG="--resume $CKPT"; echo "resuming from $CKPT"; fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OURS_CROP_TRANSFORM=1

echo "JobID=${SLURM_JOB_ID} node=$(hostname) start=$(date)"
echo "SIZE=${SIZE} SEED=${SEED} RUN=${RUN} epochs=36 OURS_CROP_TRANSFORM=${OURS_CROP_TRANSFORM}"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir "$RUN" $RESUME_ARG \
  --train-image-list "$V2/train_${SIZE}.txt" \
  --train-scanpath-ids 0 1 2 3 4 \
  --paths-per-image 5 \
  --num-epochs 36 \
  --batch-size 128 --num-workers 16 --seed "$SEED" \
  --heatmap-size 64 --heatmap-sigma 2.0 --max-seq-len 16 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --decoder-upsample nearest \
  --lr 3e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss ll --amp \
  --log-every 200 --val-batches -1 \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name "v2_${SIZE}_36ep_5p_ll_seed${SEED}"

echo "end=$(date)"
