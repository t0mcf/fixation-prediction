#!/bin/bash
#SBATCH --job-name=triangle_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

# Generator triangle: which synthetic-gaze source pretrains the model better,
# with the scanpath-length confound isolated.
#   sd5     ScanDiff-teacher corpus (unfiltered; out-of-bounds fixations are
#           clamped to the border cell by the LL objective)
#   avfull  same images, same per-image path count, real AV paths, natural len
#   avmatch same AV paths, prefix-truncated to the index-paired SD lengths
# All arms otherwise identical: crop geometry, fixed paths, 30 epochs, seed 42,
# full validation set. Checkpoint selection: fixed final epoch (the
# --parquet-path val is AV data; do not select on it).
#
#   sbatch --export=ALL,ARM=sd5     slurm/triangle_v2.sh
#   sbatch --export=ALL,ARM=avfull  slurm/triangle_v2.sh
#   sbatch --export=ALL,ARM=avmatch slurm/triangle_v2.sh

set -eo pipefail
: "${ARM:?set ARM to sd5, avfull or avmatch}"
SDD=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data
case "$ARM" in
  sd5)     TP=$SDD/scandiffgen_official_v2_avcrop_100k_unfiltered/merged.parquet ;;
  avfull)  TP=$SDD/av_fixed5_full_v2u_100k/merged.parquet ;;
  avmatch) TP=$SDD/av_fixed5_match_v2u_100k/merged.parquet ;;
  *) echo "bad ARM=$ARM" >&2; exit 2 ;;
esac
test -f "$TP" || { echo "training parquet missing: $TP (build_av_fixed5_parquets.py first?)" >&2; exit 4; }

REPO=${REPO:-$HOME/repos/fixation-prediction}
RUN="runs/triangle_${ARM}_v2u_100k"
cd "$REPO"
mkdir -p logs
if [ -e "$RUN" ]; then echo "refusing to overwrite existing run: $RUN" >&2; exit 3; fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OURS_CROP_TRANSFORM=1

mkdir -p "$RUN/code_snapshot"
cp -r src "$RUN/code_snapshot/"
git rev-parse HEAD > "$RUN/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "$RUN/code_snapshot/workdir.diff" 2>/dev/null || true
echo "JobID=${SLURM_JOB_ID} node=$(hostname) ARM=$ARM TP=$TP OURS_CROP_TRANSFORM=1 start=$(date)"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir "$RUN" \
  --paths-per-image 5 --train-parquet-path "$TP" \
  --batch-size 128 --num-epochs 30 --num-workers 16 --seed 42 \
  --heatmap-size 64 --heatmap-sigma 2.0 --max-seq-len 16 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --lr 3e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss ll --amp \
  --log-every 200 --val-batches -1 \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-name "triangle_${ARM}_v2u_100k"

echo "end=$(date)"
