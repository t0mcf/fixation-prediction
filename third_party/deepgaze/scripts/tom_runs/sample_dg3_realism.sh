#!/bin/bash
#SBATCH --job-name=dg3_sample_realism
#SBATCH --output=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/output_%A.txt
#SBATCH --error=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather/logs/job_%A/error_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=inet
# Sample scanpaths from the synthetic-trained DeepGaze III ladder (zero-shot on
# MIT1003, length-matched) so DG3 can be placed on the MultiMatch/ScanMatch axis
# next to our model and ScanDiff.
set -eo pipefail
source ~/.bashrc
conda activate praktikum
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export TORCH_HOME='/mnt/vast-nhr/home/tomcosmo.fischer/u27846/.cache/torch'
cd /mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/scanpather
mkdir -p logs/job_${SLURM_JOB_ID:-local}
SPEC=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval/mit1003_human_full.json
STIM=/mnt/vast-nhr/projects/nim00018/datasets/MIT1003/stimuli
OUT=/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval
for SCALE in 10k 50k 200k; do
  echo "=== sampling DG3-${SCALE} ==="
  python -m scanpath.sample_scanpaths_dg3 \
    --checkpoint runs/imagenet_paths/dg3_${SCALE}_5sp/scanpath/best.pth \
    --spec "$SPEC" --stimuli-dir "$STIM" \
    --out "$OUT/gen_mit1003_dg3_${SCALE}.npy" \
    --temperature 1.0
done
echo "SAMPLING DONE"
