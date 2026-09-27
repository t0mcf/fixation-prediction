#!/bin/bash
#SBATCH --job-name=sdv2_train
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=48:00:00
set -euo pipefail

: "${SIZE:?Set SIZE to 1k, 10k, 50k, 100k, or 200k}"
: "${SEED:=0}"
# FROZEN=1 trains under the authors' effective regime (BatchNorm statistics
# frozen + dropout off from epoch 1; trainer.freeze_mode_after_epoch0). The
# run directory gets a _frozen suffix so the two regimes can never be mixed up.
: "${FROZEN:=0}"
# MAX_EPOCHS override exists for the smoke test only (e.g. MAX_EPOCHS=2);
# real runs use the default 120.
: "${MAX_EPOCHS:=120}"
case "$SIZE" in 1k|10k|50k|100k|200k) ;; *) echo "Invalid SIZE=$SIZE" >&2; exit 2;; esac

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
NAME=oursynth_v2_${SIZE}
SUFFIX=""
FROZEN_OVERRIDE=""
if [ "$FROZEN" = "1" ]; then
  SUFFIX="_frozen"
  FROZEN_OVERRIDE="trainer.freeze_mode_after_epoch0=true"
fi
RUN=$SD/runs_thesis_v2/${NAME}${SUFFIX}_seed${SEED}
FEATURES=$SD/data/thesis_v2/$NAME/dinov2_base_timm_image_features
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd "$SD"

test -f "$FEATURES/features.npy"
test -f "$FEATURES/memmap_index.json"
if [ -e "$RUN/train/checkpoints/last.pt" ]; then
  echo "Fresh job refuses existing checkpoint: $RUN" >&2
  exit 3
fi

# Provenance: snapshot the code this run actually executes. The Aug-15/20
# retrain divergence could only be reconstructed from checkpoint buffers
# because no snapshot existed; never again.
mkdir -p "$RUN/code_snapshot"
cp -r src configs "$RUN/code_snapshot/"
git rev-parse HEAD > "$RUN/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "$RUN/code_snapshot/workdir.diff" 2>/dev/null || true
echo "FROZEN=$FROZEN SIZE=$SIZE SEED=$SEED RUN=$RUN"

"$PY" src/train.py \
  hydra.run.dir="$RUN/train" \
  data/train_datasets="[$NAME]" data/val_datasets="[$NAME]" data/test_datasets="[$NAME]" \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=true logger.wandb.project=scandiff-thesis-v2 logger.wandb.name="${NAME}${SUFFIX}_seed${SEED}" \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=24 data.pin_memory=true data.batch_size_train=128 data.batch_size_validation=128 \
  trainer.max_epochs="$MAX_EPOCHS" trainer.validation_every_n_epochs=5 \
  seed="$SEED" train=true test=false tags="[thesis_v2,$NAME,seed${SEED}]" \
  $FROZEN_OVERRIDE

# Spatial tripwire at the end of every training segment: cheap sampled check
# for the displaced-map failure mode (retrain B). ~2-5 min at respacing 100.
# A failure here must be LOUD but must not mask a successful training exit.
"$PY" scripts/check_v2_spatial.py --size "$SIZE" --run-dir "$RUN" \
  --respacing 100 --limit 0.005 \
  || echo "WARNING: spatial tripwire failed to run — check manually"
