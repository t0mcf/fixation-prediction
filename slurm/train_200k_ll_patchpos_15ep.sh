#!/bin/bash
#SBATCH --job-name=train_200k_ll_patchpos_15ep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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

# 200k images, NLL loss, + learned 2D patch position embeddings
# ablation against train_200k_ll_15ep: identical in scale/data, but with two
# code improvements: (1) decoder-dropout 0.0 (Dropout2d at 0.1 was zeroing entire
# feature channels in the decoder, likely too aggressive for spatial heatmap
# prediction); (2) cuDNN benchmark mode (faster conv kernel selection).
# patch_pos_embed adds nn.Embedding(256, 256) trained end-to-end, applied after the
# image adapter. tests whether explicit learned patch positions help cross-attention
# align spatial proximity between image tokens and scanpath tokens.
python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/train_200k_ll_patchpos_15ep \
  --max-images 200000 \
  --batch-size 128 \
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
  --patch-pos-embed \
  --amp \
  --log-every 200 \
  --val-batches 400 \
  --use-wandb \
  --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name train_200k_ll_patchpos_15ep

echo "end: $(date)"
echo "================================================================================"
