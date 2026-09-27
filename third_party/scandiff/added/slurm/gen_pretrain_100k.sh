#!/bin/bash
#SBATCH --job-name=sd_gen100k
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%A_%a.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet
# Phase 2 at the scale that can actually answer the question: the t90 teacher
# generates scanpaths on the oursynth_100k images. The 8k run was below the
# scale where pretraining beats scratch at all (1.791 vs 1.797 vs scratch
# 1.806 -- all noise), so it could not rank the two generators.
#
# 5 scanpaths/image, NOT 16: our ladder trains with --paths-per-image 5, so the
# 8k run threw away 11 of every 16 generated scanpaths (~69% of its GPU time).
# 100k images / 200 shards = 500 img/shard; at ~200 img/h that is ~2.5h/shard --
# short jobs backfill into scheduler gaps instead of waiting for big free blocks.
#   sbatch --array=0-199 slurm/gen_pretrain_100k.sh
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
S=${SLURM_ARRAY_TASK_ID:?array required}
OUT=data/scandiffgen_100k/shard_${S}.parquet
if [ -f "$OUT" ]; then echo "shard $S already done, skipping"; exit 0; fi
$PY scripts/generate_pretrain_scandiff.py \
  --dataset oursynth_100k \
  --sp-per-image 5 \
  --respacing 1000 \
  --shard $S --num-shards 200 \
  --out "$OUT"
echo "GEN SHARD $S DONE"
