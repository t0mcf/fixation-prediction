#!/bin/bash
#SBATCH --job-name=dg3_mit_ourfolds
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A_%a.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet
#SBATCH --mem=128G

# DG3 MIT1003 10-fold CV fine-tune on OUR seed-42 image folds (fold_split_seed),
# for the fair cross-model comparison. One fold per array task.
#   sbatch --array=0-9 --export=ALL,COND=scratch scripts/tom_runs/dg3_mit1003_ourfolds.sh
#   sbatch --array=0-9 --export=ALL,COND=100k    scripts/tom_runs/dg3_mit1003_ourfolds.sh
#   sbatch --array=0-9 --export=ALL,COND=200k    scripts/tom_runs/dg3_mit1003_ourfolds.sh
# scratch: full 3-stage from random init (no SALICON, matching exp16's no-pretrain)
# 100k/200k: init from imagenet scanpath checkpoint (stage 1 skipped)

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
cd "${REPO_ROOT}"

COND="${COND:?set COND=scratch|100k|200k}"
FOLD="${SLURM_ARRAY_TASK_ID:?array task id required}"

case "$COND" in
  scratch) PRETRAIN_ARGS="--spatial_pretrain_path none" ;;
  # exp16-faithful pathway: imagenet SPATIAL checkpoint -> full 3-stage curriculum
  # (stage 1 MIT spatial finetune INCLUDED). The earlier scanpath_pretrain_path
  # variant skipped stage 1 and handicapped the pretrained conditions
  # (those runs archived as *_skipstage1).
  100k)    PRETRAIN_ARGS="--spatial_pretrain_path runs/imagenet_paths/dg3_100k_5sp/spatial/final.pth" ;;
  200k)    PRETRAIN_ARGS="--spatial_pretrain_path runs/imagenet_paths/dg3_200k_5sp/spatial/final.pth" ;;
  # backbone-swap ablation: DG3 readout on frozen DINOv2 (our backbone),
  # full 3-stage from random readout init — compare vs scratch (DenseNet)
  dinov2scratch) PRETRAIN_ARGS="--spatial_pretrain_path none --backbone dinov2" ;;
  *) echo "unknown COND=$COND"; exit 1 ;;
esac

echo "COND=$COND fold=$FOLD start=$(date)"
python -m scanpath.train_deepgaze \
    --config scanpath/configs/mit1003.yaml \
    $PRETRAIN_ARGS ${EXTRA_ARGS:-} \
    --crossval_folds 10 \
    --fold_only "$FOLD" \
    --fold_split_seed 42 \
    --out_dir "runs/dg3_mit1003_ourfolds_${COND}" \
    --run_name "dg3_mit1003_ourfolds_${COND}_fold${FOLD}" \
    --use_wandb false
echo "end=$(date)"
