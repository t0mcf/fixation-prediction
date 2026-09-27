#!/bin/bash
#SBATCH --job-name=sd_ft_mit
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=inet

# ScanDiff Arm A: MIT1003-only training under the FROZEN regime (the regime
# measured better on human data — R1; the regime flag replaces the authors'
# evaluator leak explicitly). Two conditions:
#   COND=scratch  from-scratch on the authors' MIT1003 train split
#   COND=av100k   init_from the AV-pretrained 100k ladder rung (train-mode
#                 pretraining — measured better in-domain, gate 2026-08-24;
#                 BN recalibrates for one epoch on human data, then freezes)
# Selection: fixed endpoint primary; best-val via offline scoring of the
# every-5-epoch checkpoints on the authors' validation split.
#
#   sbatch --export=ALL,COND=scratch,SEED=0 slurm/ft_scandiff_mit.sh

set -euo pipefail
: "${COND:?set COND (scratch|av1k|av10k|av50k|av100k|av200k)}"
: "${SEED:=0}"
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/paper_reproduction/data
: "${INIT_CKPT:=last.pt}"   # which pretraining checkpoint to init from; != last.pt -> run suffix _bvinit
SUF=""; [ "$INIT_CKPT" != "last.pt" ] && SUF="_bvinit"
RUN=$SD/runs_thesis_v2/ft_mit_${COND}_seed${SEED}${SUF}
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
export WANDB_MODE=offline
cd "$SD"

INIT_OVERRIDE=""
if [ "${COND#av}" != "$COND" ]; then
  INIT=$SD/runs_thesis_v2/oursynth_v2_${COND#av}_seed0/train/checkpoints/$INIT_CKPT
  test -f "$INIT" || { echo "missing init: $INIT" >&2; exit 4; }
  INIT_OVERRIDE="init_from=$INIT"
fi
if [ -e "$RUN/train/checkpoints/last.pt" ]; then
  echo "refusing existing run: $RUN" >&2; exit 3
fi

mkdir -p "$RUN/code_snapshot" "$SD/slurm/logs"
cp -r src configs "$RUN/code_snapshot/"
git rev-parse HEAD > "$RUN/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "$RUN/code_snapshot/workdir.diff" 2>/dev/null || true
echo "COND=$COND SEED=$SEED INIT=${INIT_OVERRIDE:-scratch}"

"$PY" src/train.py \
  hydra.run.dir="$RUN/train" \
  data/train_datasets="[mit1003]" data/val_datasets="[mit1003]" data/test_datasets="[mit1003]" \
  data.train_datasets.mit1003.root_path="$DATA" \
  data.val_datasets.mit1003.root_path="$DATA" \
  data.test_datasets.mit1003.root_path="$DATA" \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=true logger.wandb.project=scandiff-thesis-v2 \
  logger.wandb.name="ft_mit_${COND}_seed${SEED}${SUF}" \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=8 data.batch_size_train=128 data.batch_size_validation=128 \
  trainer.max_epochs=200 trainer.validation_every_n_epochs=10 \
  trainer.freeze_mode_after_epoch0=true \
  seed="$SEED" train=true test=false \
  tags="[thesis_v2,ft_mit,$COND,seed${SEED}]" \
  $INIT_OVERRIDE
echo "end=$(date)"
