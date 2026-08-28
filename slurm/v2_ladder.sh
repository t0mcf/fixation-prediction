#!/bin/bash
#SBATCH --job-name=v2_ladder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet

# Final in-domain scaling ladder for the thesis, under protocol v2.
#
# Differences from the pilot ladder in docs/report_tables/scaling_ladder.csv,
# each of which was a real defect there:
#
#   images     protocol-v2 lists instead of `unique()[:N]` in parquet order.
#              The parquet is sorted by class, so the old 1k run saw ONE
#              ImageNet class and the old 10k run eight. v2 gives 634 classes
#              at 1k and all 1000 from 10k up, matching DG3's own subsets.
#   scanpaths  fixed IDs 0-4, the same five every epoch. The old sampler drew a
#              fresh 5-of-16 per epoch, so the model effectively saw all
#              sixteen -- roughly 3x the distinct scanpaths DeepGaze III and
#              ScanDiff get, in the model's own favour.
#   geometry   Resize(256)+CenterCrop(224), matching the generator's own frame.
#              The old ladder resized directly to 224, so a coordinate pointed
#              at a different place for generator and model.
#   epochs     18 at every scale. The old ladder ran 30/30/30/30/20/15/12, and
#              since the cosine schedule spans num_epochs * steps_per_epoch,
#              every rung had a different learning-rate trajectory -- data
#              scale and optimisation were confounded. Every observed optimum
#              in the pilot fell on or before epoch 12, so 18 leaves headroom.
#   validation full 5000-image set, not the first 400 batches. Checkpoint
#              selection previously ran on ~64% of a class-sorted prefix.
#
#   SIZE=1k|10k|50k|100k|200k|400k|800k   SEED defaults to 42
#   sbatch --export=ALL,SIZE=1k --time=02:00:00 slurm/v2_ladder.sh

set -eo pipefail
: "${SIZE:?set SIZE to one of 1k 10k 50k 100k 200k 400k 800k}"
: "${SEED:=42}"
case "$SIZE" in 1k|10k|50k|100k|200k|400k|800k) ;; *) echo "bad SIZE=$SIZE" >&2; exit 2;; esac

REPO=${REPO:-$HOME/repos/fixation-prediction}
V2=/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av
RUN="runs/v2_${SIZE}_5p_ll_seed${SEED}"

cd "$REPO"
mkdir -p logs
# a fresh run must never continue an old one by accident
if [ -e "$RUN/checkpoints/latest.pt" ]; then
  echo "refusing to start on top of an existing run: $RUN" >&2
  exit 3
fi
test -f "$V2/train_${SIZE}.txt"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# geometry is part of the protocol, not a default -- set it explicitly
export OURS_CROP_TRANSFORM=1

echo "JobID=${SLURM_JOB_ID} node=$(hostname) start=$(date)"
echo "SIZE=${SIZE} SEED=${SEED} RUN=${RUN} OURS_CROP_TRANSFORM=${OURS_CROP_TRANSFORM}"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir "$RUN" \
  --train-image-list "$V2/train_${SIZE}.txt" \
  --train-scanpath-ids 0 1 2 3 4 \
  --paths-per-image 5 \
  --num-epochs 18 \
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
  --wandb-name "v2_${SIZE}_5p_ll_seed${SEED}"

echo "end=$(date)"
