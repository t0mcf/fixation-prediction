#!/bin/bash
#SBATCH --job-name=train_200k_ll_bidir_bs64_15ep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

set -eo pipefail

mkdir -p logs

echo "================================================================================"
echo "JobID = ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "start: $(date)"
echo "================================================================================"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

# 200k images, NLL loss, bidirectional co-attention fusion, cosine LR, batch size 64
# retry of train_200k_ll_bidir_15ep which OOM'd at bs128. batch size halved to fit
# the doubled memory footprint of bidirectional fusion (both attention directions
# computed simultaneously, B*T tiled batch goes from 1920 to 960).
python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/train_200k_ll_bidir_bs64_15ep \
  --max-images 200000 \
  --batch-size 64 \
  --num-epochs 15 \
  --num-workers 16 \
  --seed 42 \
  --paths-per-image 16 \
  --heatmap-size 64 \
  --heatmap-sigma 2.0 \
  --max-seq-len 16 \
  --visual-encoder dinov2_base \
  --d-model 256 \
  --n-heads 8 \
  --scanpath-layers 2 \
  --fusion-layers 4 \
  --dropout 0.1 \
  --decoder-dropout 0.0 \
  --decoder-hidden-channels 256 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --encoder-lr-scale 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --warmup-fraction 0.05 \
  --loss ll \
  --bidirectional-fusion \
  --amp \
  --log-every 200 \
  --val-batches 400 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name train_200k_ll_bidir_bs64_15ep

echo "end: $(date)"
echo "================================================================================"
