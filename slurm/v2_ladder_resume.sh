#!/bin/bash
#SBATCH --job-name=v2_resume
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

# Continue a run started by slurm/v2_ladder.sh after it hit its walltime.
#
# Every argument below is identical to the fresh script on purpose. The cosine
# schedule spans num_epochs * steps_per_epoch, so changing either the epoch
# budget or the data here would give this rung a different learning-rate curve
# than the others -- which is exactly the confound the uniform 18-epoch budget
# exists to remove. train.py restores model, optimizer, scheduler and scaler
# state from the checkpoint, so the schedule picks up where it left off.
#
# Chain these with afterany, not afterok: a run that ends by hitting the
# walltime exits as TIMEOUT, which does not satisfy afterok.
#
# If the run already reached its last epoch, the loop has nothing to do and the
# job exits quickly, so an unused link in the chain is cheap.
#
#   sbatch --export=ALL,SIZE=800k --dependency=afterany:<prev> slurm/v2_ladder_resume.sh

set -eo pipefail
: "${SIZE:?set SIZE to one of 1k 10k 50k 100k 200k 400k 800k}"
: "${SEED:=42}"

REPO=${REPO:-$HOME/repos/fixation-prediction}
V2=/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av
RUN="runs/v2_${SIZE}_5p_ll_seed${SEED}"
CKPT="$RUN/checkpoints/latest.pt"

cd "$REPO"
mkdir -p logs
test -f "$CKPT" || { echo "nothing to resume: $CKPT missing" >&2; exit 3; }

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OURS_CROP_TRANSFORM=1

echo "JobID=${SLURM_JOB_ID} node=$(hostname) start=$(date)"
echo "resuming ${RUN} from ${CKPT}"

python -m src.training.train \
  --parquet-path /mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet \
  --imagenet-root /mnt/vast-nhr/projects/nim00018/datasets/ImageNet \
  --output-dir "$RUN" \
  --resume "$CKPT" \
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
