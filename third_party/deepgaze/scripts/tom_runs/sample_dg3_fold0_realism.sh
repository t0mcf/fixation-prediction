#!/bin/bash
#SBATCH --job-name=dg3_sample_fold0
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
# Sample scanpaths from FINE-TUNED (not zero-shot) DG3 fold-0 checkpoints --
# scratch-on-MIT1003 and 100k-pretrained-then-finetuned -- on the same 101
# held-out images ScanDiff's t90/pilot were scored on. Companion to
# fixation-prediction/slurm/sample_fold0_realism.sh (ours side).
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}
SPEC=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval/mit1003_human_fold0.json
STIM=/mnt/vast-nhr/projects/nim00018/datasets/MIT1003/stimuli
OUT=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval
B=/mnt/vast-nhr/projects/nim00018/tom/scanpather-artifacts

echo "=== sampling DG3-scratch (fold0) ==="
python -m scanpath.sample_scanpaths_dg3 \
  --checkpoint "$B/runs/dg3_mit1003_ourfolds_scratch/MIT1003_scanpath_partially_frozen_saliency_network/crossval-10-0/best.pth" \
  --spec "$SPEC" --stimuli-dir "$STIM" \
  --out "$OUT/gen_fold0_dg3-scratch.npy" \
  --temperature 1.0

echo "=== sampling DG3-100k-finetuned (fold0) ==="
python -m scanpath.sample_scanpaths_dg3 \
  --checkpoint "$B/runs/dg3_mit1003_ourfolds_100k/MIT1003_scanpath_partially_frozen_saliency_network/crossval-10-0/best.pth" \
  --spec "$SPEC" --stimuli-dir "$STIM" \
  --out "$OUT/gen_fold0_dg3-100k.npy" \
  --temperature 1.0

echo "SAMPLING DONE"
