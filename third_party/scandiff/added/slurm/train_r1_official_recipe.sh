#!/bin/bash
#SBATCH --job-name=sd_r1_official
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

# R1 — reproduction of the authors' ACTUAL recipe (jobs/train.sh), the run
# that proves or refutes the BatchNorm/eval-mode finding end-to-end.
#
# Faithful to jobs/train.sh: dataset order [coco_freeview,mit1003], val
# [mit1003], test [coco_freeview], 200 epochs, batch 128, num_workers 8,
# checkpoint every 10 epochs, spaced diffusion @1000 steps, train=true
# test=true (the evaluator is what triggers the authors' eval-mode leak).
#
# DOCUMENTED deviations, both deliberate:
#   1. seed=0 — the authors never seeded (their train.py imports
#      seed_everything and never calls it); without a seed the run is not
#      reproducible even in principle.
#   2. validation/test cadence 1000 instead of 10 — the evaluator then fires
#      exactly ONCE, at epoch 0. That single firing is what freezes the mode
#      (the regime-forming event); the metric TRAJECTORY is scored offline on
#      the every-10-epoch checkpoints instead (score_r1_checkpoints.sh),
#      because 40 in-loop sampling passes would cost ~4-7 days of walltime
#      for numbers the offline path computes identically.
#
# SUCCESS CRITERIA: (a) final checkpoint's encoder/decoder num_batches_tracked
# == 318 (one epoch); (b) some scored epoch lands near the released
# checkpoint's MIT1003 numbers (MM KLD ~0.040, SM KLD ~0.041, SS KLD ~0.047).

set -euo pipefail
: "${SEED:=0}"
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/paper_reproduction/data
RUN=$SD/paper_reproduction/r1_official_recipe_seed${SEED}
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
export WANDB_MODE=offline
cd "$SD"

if [ -e "$RUN/train/checkpoints/last.pt" ]; then
  echo "Fresh job refuses existing checkpoint: $RUN" >&2
  exit 3
fi

mkdir -p "$RUN/code_snapshot" "$SD/paper_reproduction/logs"
cp -r src configs jobs "$RUN/code_snapshot/"
git rev-parse HEAD > "$RUN/code_snapshot/commit.txt" 2>/dev/null || true
git diff > "$RUN/code_snapshot/workdir.diff" 2>/dev/null || true

"$PY" src/train.py \
  hydra.run.dir="$RUN/train" \
  data/train_datasets="[coco_freeview,mit1003]" \
  data/val_datasets="[mit1003]" \
  data/test_datasets="[coco_freeview]" \
  data.train_datasets.coco_freeview.root_path="$DATA" \
  data.train_datasets.mit1003.root_path="$DATA" \
  data.val_datasets.mit1003.root_path="$DATA" \
  data.test_datasets.coco_freeview.root_path="$DATA" \
  trainer=gpu diffusion.num_timesteps=1000 \
  data.num_workers=8 callbacks=default \
  logger=wandb logger.wandb.offline=true \
  logger.wandb.project=scandiff-thesis-v2 logger.wandb.name=r1_official_recipe_seed${SEED} \
  tags="[r1,official_recipe,seed0]" \
  trainer.validation_every_n_epochs=1000 \
  trainer.test_every_n_epochs=1000 \
  trainer.max_epochs=200 \
  train=true test=true \
  evaluation.data_to_extract="['preds','metrics','qualitatives']" \
  evaluation.metrics_to_compute="['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time']" \
  diffusion_class=spaced_diffusion \
  callbacks.model_checkpoint.every_n_epochs=10 \
  ckpt_path=null seed=$SEED

# Immediately verify the regime evidence: the BN counters of the finished run.
"$PY" - <<EOF
import torch
ck = torch.load("paper_reproduction/r1_official_recipe_seed${SEED}/train/checkpoints/last.pt",
                map_location="cpu")
sd = ck["model"]
for k in ("encoder.fcs.1.num_batches_tracked", "decoder.fcs.1.num_batches_tracked"):
    print(f"R1 REGIME CHECK  {k} = {int(sd[k])}   (released checkpoint: 318)")
EOF
