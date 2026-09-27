#!/bin/bash
#SBATCH --job-name=sd_resume
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Usage: sbatch --export=ALL,SIZE=50k,EPOCHS=120 slurm/resume_scaling.sh
# Resumes from the newest checkpoint in runs_ours/train_<SIZE>/train/checkpoints.
#
# Throughput fix (replaces the earlier /local NVMe-staging approach):
#   Each sample is one contiguous 2.1MB read from the Lustre feature memmap.
#   The original slowness (0.33 batch/s) was latency, not bandwidth: with only
#   4 workers too few reads were ever in flight (~45MB/s vs Lustre's GB/s).
#   Fix: just parallelise the reads -- originally 12 workers (12 cpus),
#   persistent across epochs (see unified_datamodule.py). No node-local
#   scratch to fill or collide on, and it helps every scale equally, 200k
#   included.
#
#   2026-08-10: 12 workers was STILL not enough -- live nvidia-smi on the
#   200k run showed bursty GPU util (66%, 68%, then 0%, 0%, 0% -- stalling on
#   data between batches most of the time), while the node itself had 128
#   CPUs and 417G free RAM sitting idle (job was only using 12/128 cores).
#   Bumped to 24 workers + enabled pin_memory (was false, the datamodule
#   default -- unified_datamodule.yaml exposes it as data.pin_memory).
#
#   Memory: /dev/shm (how workers hand batches back) and the memmap page cache
#   both count against the job's memory cgroup. The earlier 16-worker attempt
#   crashed with a shared-memory Bus error purely because it kept the old
#   --mem=64G; scaled memory up proportionally with the new worker count
#   (256G is still half of a 512G node, not a whole-node hog, and this node
#   had far more than that free).

set -e

: "${SIZE:?Set SIZE (e.g. 50k, 100k, 200k)}"
: "${EPOCHS:=120}"
: "${NUM_WORKERS:=24}"

export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface
export WANDB_MODE=offline
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_${SIZE}
CKPT_DIR=$RUN/train/checkpoints

# newest checkpoint by mtime (last.pt is usually newer than the newest epoch_*.pt)
CKPT_PATH=$(ls -1t "$CKPT_DIR"/last.pt "$CKPT_DIR"/epoch_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
  echo "No checkpoint found in $CKPT_DIR" >&2
  exit 1
fi
echo "Resuming oursynth_${SIZE} from $CKPT_PATH, target ${EPOCHS} epochs, ${NUM_WORKERS} workers"

$PY src/train.py \
  hydra.run.dir=$RUN/train \
  data/train_datasets=[oursynth_${SIZE}] \
  data/val_datasets=[oursynth_${SIZE}] \
  data/test_datasets=[oursynth_${SIZE}] \
  trainer=gpu callbacks=default logger=wandb \
  logger.wandb.offline=True \
  logger.wandb.project=scandiff-scaling \
  logger.wandb.name=oursynth_${SIZE} \
  diffusion.num_timesteps=1000 diffusion_class=spaced_diffusion \
  data.num_workers=${NUM_WORKERS} data.pin_memory=true data.batch_size_validation=128 \
  trainer.max_epochs=${EPOCHS} trainer.validation_every_n_epochs=5 \
  ckpt_path=$CKPT_PATH \
  train=true test=false \
  tags=[oursynth_${SIZE}]

echo "Done: $RUN"
