#!/bin/bash
#SBATCH --job-name=ablate_loss
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --constraint=inet

# Loss ablation, redone on the canonical split.
#
# Why redone: the previously quoted arms came from two different eras --
# KL sigma=1.0 and sigma=4.0 were trained 2026-08-08 (canonical split, clean),
# but KL sigma=2.0 and the NLL baseline date from 2026-05-28 and were trained
# on the pre-canonical split, whose training pool overlaps ~16% of the
# canonical val set (see the loss-comparability note in the report). Comparing an inflated NLL
# number against a clean KL number is not a valid ablation.
#
# This array trains the three KL arms at EXACTLY the configuration of
# runs/scale_50k_5p_ll, which is canonical-split, NLL, seed 42, 30 epochs, and
# therefore serves as the fourth (NLL) arm without retraining. Only --loss and
# --heatmap-sigma differ between the four arms.
#
# 30 epochs is not needed for convergence (scale_50k_5p_ll peaks at epoch 5) but
# matches the reused NLL arm exactly and leaves ample headroom in case KL
# converges more slowly.
#   sbatch --array=0-2 slurm/ablate_loss_50k.sh
set -eo pipefail
mkdir -p logs
SIGMAS=(1.0 2.0 4.0)
S=${SIGMAS[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=$(echo "$S" | tr -d '.')
echo "JobID=${SLURM_JOB_ID} sigma=$S node=$(hostname) start=$(date)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/ablate_loss_50k_kl_sigma${TAG} \
  --max-images 50000 --paths-per-image 5 \
  --batch-size 128 --num-epochs 30 --num-workers 16 --seed 42 \
  --heatmap-size 64 --heatmap-sigma ${S} --max-seq-len 16 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --lr 3e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss kl --amp \
  --log-every 200 --val-batches 400 \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen --wandb-name ablate_loss_50k_kl_sigma${TAG}
echo "end=$(date)"
