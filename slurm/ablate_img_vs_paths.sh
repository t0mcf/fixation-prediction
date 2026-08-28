#!/bin/bash
#SBATCH --job-name=ablate_ivp
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --constraint=inet

# Images versus scanpaths per image, at a FIXED total number of scanpaths.
#
# Why redone rather than reusing train_200k_ll_15ep / train_400k_8p_ll_20ep /
# train_800k_4p_nopatchpos_20ep: those three predate the canonical split, and
# their contamination of the canonical val set grows with exactly the variable
# under test -- 16% of val inside the 200k pool, 32% inside 400k, 63% inside
# 800k (re-measured 2026-08-15). The arm with the
# most images was also the most contaminated, so "more images is better" could
# not be separated from "more images already seen in training". Those runs also
# had unequal epoch budgets (15/20/20).
#
# Here every arm trains on 400k scanpaths for 30 epochs, so all three take
# exactly 93,750 optimizer steps; only the split between number of images and
# scanpaths per image differs.
#   sbatch --array=0-2 slurm/ablate_img_vs_paths.sh
set -eo pipefail
mkdir -p logs
IMAGES=(25000 50000 100000)
PATHS=(16 8 4)
N=${IMAGES[${SLURM_ARRAY_TASK_ID:-0}]}
P=${PATHS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "JobID=${SLURM_JOB_ID} images=$N paths=$P node=$(hostname) start=$(date)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/ablate_ivp_${N}img_${P}p \
  --max-images ${N} --paths-per-image ${P} \
  --batch-size 128 --num-epochs 30 --num-workers 16 --seed 42 \
  --heatmap-size 64 --heatmap-sigma 2.0 --max-seq-len 16 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --lr 3e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss ll --amp \
  --log-every 200 --val-batches 400 \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen --wandb-name ablate_ivp_${N}img_${P}p
echo "end=$(date)"
