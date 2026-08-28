#!/bin/bash
#SBATCH --job-name=ablate_dec
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --constraint=inet

# Heatmap-decoder upsampling ablation at the 10k rung.
# Identical to slurm/scale_10k_5p_ll.sh in EVERY respect except --decoder-upsample,
# and all three arms are run with the same code version so they are comparable
# (the existing scale_10k_5p_ll result predates the flag and is not reused).
# Quality: canonical-val NSS/LL/AUC. Artifacts: heatmaps from the transposed arm.
# Timing lives in scripts/benchmark_decoder.py, not here -- wall clock on a
# shared node is not a clean speed measurement.
#   sbatch --array=0-2 slurm/ablate_decoder_10k.sh
set -eo pipefail
mkdir -p logs
MODES=(nearest bilinear transposed)
M=${MODES[${SLURM_ARRAY_TASK_ID:-0}]}
echo "JobID=${SLURM_JOB_ID} mode=$M node=$(hostname) start=$(date)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir runs/ablate_decoder_10k_${M} \
  --max-images 10000 --paths-per-image 5 \
  --decoder-upsample ${M} \
  --batch-size 128 --num-epochs 30 --num-workers 16 --seed 42 \
  --heatmap-size 64 --heatmap-sigma 2.0 --max-seq-len 16 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --lr 3e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss ll --amp \
  --log-every 200 --val-batches 400 \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen --wandb-name ablate_decoder_10k_${M}
echo "end=$(date)"
