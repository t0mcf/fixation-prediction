#!/bin/bash
#SBATCH --job-name=nss_ceiling_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Generator ceiling on the full canonical validation set (5000 images, all 16
# scanpaths/image; identical protocol to runs/fullval_eval/ceiling_fullval.log),
# additionally lifted to DeepGaze III's 224 grid so the ceiling's LL is on the
# same grid as indomain_ours_v2.csv:ll_dg3grid and indomain_dg3_v2.csv:ll.
# Walltime: same request as the archived fullval run (no sacct record survives);
# the lift adds one bilinear resize per step.

set -eo pipefail
mkdir -p logs runs/fullval_eval_test

echo "JobID=${SLURM_JOB_ID} node=$(hostname) start=$(date)"
echo "protocol: protocol test_10k, 10000 images (TEST DAY), all 16 scanpaths/image, lift 224"

PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python

$PY scripts/nss_ceiling.py \
  --batch-size 16 \
  --max-scanpaths-per-image 0 \
  --eval-resolution 224 --image-list /mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av/test_10k.txt | tee runs/fullval_eval_test/ceiling_fullval_224.log

echo "end=$(date)"
