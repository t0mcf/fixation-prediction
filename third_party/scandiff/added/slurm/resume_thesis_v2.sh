#!/bin/bash
#SBATCH --job-name=sdv2_resume
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
case "$SIZE" in 1k|10k|50k|100k|200k) ;; *) echo "Invalid SIZE=$SIZE" >&2; exit 2;; esac

SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
NAME=oursynth_v2_${SIZE}
RUN=$SD/runs_thesis_v2/${NAME}_seed${SEED}
CKPT_DIR=$RUN/train/checkpoints
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd "$SD"
CKPT=$("$PY" tools/select_resume_checkpoint.py "$CKPT_DIR")
echo "Resuming $NAME from $CKPT"

"$PY" src/train.py \
  hydra.run.dir="$RUN/train" \
  data/train_datasets="[$NAME]" data/val_datasets="[$NAME]" data/test_datasets="[$NAME]" \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=true logger.wandb.project=scandiff-thesis-v2 logger.wandb.name="${NAME}_seed${SEED}" \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=24 data.pin_memory=true data.batch_size_train=128 data.batch_size_validation=128 \
  trainer.max_epochs=120 trainer.validation_every_n_epochs=5 \
  seed="$SEED" ckpt_path="$CKPT" train=true test=false tags="[thesis_v2,$NAME,seed${SEED}]"
