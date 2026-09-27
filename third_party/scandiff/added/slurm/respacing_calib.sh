#!/bin/bash
#SBATCH --job-name=sd_respace_calib
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet
# Diffusion-step calibration for ScanDiff generation, in the same spirit as the
# epoch sweep: does sampling with fewer diffusion steps change the generations
# enough to matter for pretraining? Regenerates shard 0 (same 500 images as the
# existing respacing-1000 data/scandiffgen_100k/shard_0.parquet) at reduced
# respacings, so the comparison is paired on identical images. If the §4 collapse
# statistics are stable down to some step count, the full generation can use that
# and run ~10x faster -- helping the 200k scaling test AND the temperature sweep.
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
mkdir -p data/respace_calib
for R in 50 100 250; do
  OUT=data/respace_calib/shard0_respace${R}.parquet
  [ -f "$OUT" ] && { echo "respace $R done, skip"; continue; }
  echo "=== generating shard 0 at respacing $R ==="
  /usr/bin/time -v $PY scripts/generate_pretrain_scandiff.py \
    --dataset oursynth_100k --sp-per-image 5 --respacing $R \
    --shard 0 --num-shards 200 --out "$OUT" 2>&1 | grep -E "wall clock|Elapsed" || true
done
echo RESPACE_CALIB_DONE
