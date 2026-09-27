#!/bin/bash
#SBATCH --job-name=ft_authsplit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet

# MIT1003 fine-tuning on the ScanDiff authors' fixed splits (leakage-proof:
# the teacher never saw the val/test images). Replaces the earlier 10-fold CV.
# Hyperparameters identical for every arm.
#
#   sbatch --export=ALL,COND=scratch,SEED=42  slurm/ft_authsplit.sh
#   COND: scratch | sd5 | avfull | avmatch     SEED: 42 | 43

set -eo pipefail
mkdir -p logs
: "${COND:?set COND (scratch|sd5|avfull|avmatch)}"
: "${SEED:=42}"

case "$COND" in
  scratch) INIT="" ;;
  sd5)     INIT="runs/triangle_sd5_v2u_100k/checkpoints/final.pt" ;;
  avfull)  INIT="runs/triangle_avfull_v2u_100k/checkpoints/final.pt" ;;
  avmatch) INIT="runs/triangle_avmatch_v2u_100k/checkpoints/final.pt" ;;
  shuffled) INIT="runs/triangle_shuffled_v2u_100k/checkpoints/final.pt" ;;
  # pretraining-SCALE transfer axis: the v2 ladder checkpoints (18-epoch
  # pretraining protocol, vs the triangle's 30 — two documented axes).
  lad1k|lad10k|lad50k|lad100k|lad200k|lad400k|lad800k)
    INIT="runs/v2_${COND#lad}_5p_ll_seed42/checkpoints/final.pt" ;;
  # init-rule robustness control: identical to lad*, but initialised from the
  # best-val pretraining checkpoint. At small scales final.pt is overfit
  # in-domain while best_val is not, so the final-init ladder could steepen
  # the transfer trend; this arm measures that.
  bv1k|bv10k|bv50k|bv100k|bv200k|bv400k|bv800k)
    INIT="runs/v2_${COND#bv}_5p_ll_seed42/checkpoints/best_val_ll.pt" ;;
  *) echo "unknown COND=$COND" >&2; exit 1 ;;
esac
if [ -n "$INIT" ]; then test -f "$INIT" || { echo "missing init: $INIT" >&2; exit 4; }; fi
RUN="runs/ft_authsplit_${COND}_seed${SEED}"
if [ -e "$RUN" ]; then echo "refusing to overwrite: $RUN" >&2; exit 3; fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate praktikum
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN/code_snapshot"
cp -r src "$RUN/code_snapshot/"
git rev-parse HEAD > "$RUN/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "$RUN/code_snapshot/workdir.diff" 2>/dev/null || true
echo "JobID=${SLURM_JOB_ID} COND=$COND SEED=$SEED INIT=${INIT:-scratch} start=$(date)"

INIT_ARG=""
[ -n "$INIT" ] && INIT_ARG="--init-from $INIT"

python -m src.training.train \
  --dataset mit1003 \
  --mit-split-dir /mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003 \
  --output-dir "$RUN" \
  --batch-size 32 --num-epochs 30 --num-workers 8 --seed "$SEED" \
  --max-seq-len 16 --heatmap-size 64 --heatmap-sigma 2.0 \
  --visual-encoder dinov2_base --d-model 256 --n-heads 8 \
  --scanpath-layers 2 --fusion-layers 4 \
  --dropout 0.1 --decoder-dropout 0.0 --decoder-hidden-channels 256 \
  --lr 1e-4 --lr-schedule cosine --weight-decay 0.01 \
  --grad-clip 1.0 --warmup-fraction 0.05 --loss ll --amp \
  --log-every 50 --val-batches -1 \
  $INIT_ARG \
  --use-wandb --wandb-project fixation-prediction \
  --wandb-entity fischer-tom-university-of-goettingen \
  --wandb-group "ft_authsplit_${COND}" \
  --wandb-name "ft_authsplit_${COND}_seed${SEED}"
echo "end=$(date)"
