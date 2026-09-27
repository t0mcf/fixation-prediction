#!/bin/bash
#SBATCH --job-name=dg3_mit_fixedsplit
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=inet
#SBATCH --mem=128G

# DG3 MIT1003 fine-tune on the FIXED authors' splits (702 train / 150 val),
# replacing the retired fold-CV protocol. One run per condition x replicate.
# Walltime: one CV fold (900 train images) measured 9.7 h on A100; the fixed
# split is smaller, 9.7 x 1.7 buffer ~ 16.5 h, rounded up to 24 h for the
# occasional slow node.
#   sbatch --export=ALL,COND=scratch,REP=0 scripts/tom_runs/dg3_mit1003_fixedsplit.sh
#   COND: scratch | salicon | 1k | 10k | 50k | 100k | 200k | full1k | full10k | full50k | full100k | full200k
# scratch: full 3-stage from random init
# salicon: SALICON spatial checkpoint (SALICON_DIR env) -> full 3-stage
# Nk:      synthetic spatial checkpoint -> full 3-stage (exp16-faithful pathway;
#          spatial-only transfer — secondary since 2026-09-03)
# fullNk:  scanpath-MODULE transfer (primary since 2026-09-03): stage 1 from the
#          synthetic spatial checkpoint as above, stage 2 additionally seeded
#          with scanpath_network.* + history keys of the synthetic scanpath
#          checkpoint (MIT stage-1 weights overlaid). Stages 1-3 identical to
#          every other arm. CKPT_NAME (default best.pth) selects the synthetic
#          stage-2 checkpoint; use the val-selected one for 50k/100k.

set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather"
SPLITS="/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003"
mkdir -p "${REPO_ROOT}/logs/job_${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"

COND="${COND:?set COND=scratch|salicon|1k|10k|50k|100k|200k}"
REP="${REP:?set REP=0|1}"

case "$COND" in
  scratch) PRETRAIN_ARGS="--spatial_pretrain_path none" ;;
  salicon) PRETRAIN_ARGS="--spatial_pretrain_path ${SALICON_DIR:?set SALICON_DIR for COND=salicon}" ;;
  1k|10k|50k|100k|200k)
    PRETRAIN_ARGS="--spatial_pretrain_path runs/imagenet_paths/dg3_${COND}_5sp/spatial/final.pth" ;;
  full1k|full10k|full50k|full100k|full200k)
    SZ=${COND#full}
    SP="runs/imagenet_paths/dg3_${SZ}_5sp/spatial/final.pth"
    SC="runs/imagenet_paths/dg3_${SZ}_5sp/scanpath/${CKPT_NAME:-best.pth}"
    test -f "$SP" || { echo "missing $SP" >&2; exit 4; }
    test -f "$SC" || { echo "missing $SC" >&2; exit 4; }
    PRETRAIN_ARGS="--spatial_pretrain_path $SP --scanpath_module_pretrain_path $SC" ;;
  *) echo "unknown COND=$COND"; exit 1 ;;
esac

echo "COND=$COND rep=$REP start=$(date)"
python -m scanpath.train_deepgaze \
    --config scanpath/configs/mit1003.yaml \
    $PRETRAIN_ARGS ${EXTRA_ARGS:-} \
    --fixed_split_dir "$SPLITS" \
    --out_dir "runs/dg3_mit1003_fixedsplit_${COND}_rep${REP}" \
    --run_name "dg3_mit1003_fixedsplit_${COND}_rep${REP}" \
    --use_wandb false
echo "end=$(date)"
