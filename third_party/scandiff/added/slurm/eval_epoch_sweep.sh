#!/bin/bash
#SBATCH --job-name=sd_epoch_sweep
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=inet
# Where does ScanDiff training actually saturate? Evaluate ONE scale's saved
# epoch_N.pt checkpoints on its own in-domain val set with the real MM/SM+KLD
# metrics, instead of guessing from the (very noisy) diffusion validation loss.
#
# Motivation: the 400k run's feasibility hinges on the epoch budget --
# 120 ep = 15 d, 90 ep = 11 d, 60 ep = 7.5 d on one GPU (multi-GPU is NOT
# available: ScanDiff's custom src/trainer.py has no distributed support, the
# shipped configs/trainer/ddp.yaml is dead Lightning-template config).
# Binned val loss suggested saturation ~ep90, a 2-point MM/SM comparison
# suggested ~ep58 -- this settles it on the metric we actually report.
#
# NOTE: evaluates the VALIDATION split, not test -- oursynth_*_fixations_test.json
# is empty (0 records) for every ladder scale, which made the first attempt die in
# perform_kld with "zero-size array to reduction operation minimum". Hence the
# separate configs/data/test_datasets/oursynth_<scale>_valsplit.yaml.
#
# 2026-08-12: timed out 3x at the full 25k val images. trainer.limit_test_batches
# =0.08 evaluates ~2000 images -- plenty for stable MM/SM/KLD, runs in <30min.
#   sbatch --export=ALL,SCALE=100k,EPOCH=60 slurm/eval_epoch_sweep.sh
set -e
: "${SCALE:?set SCALE (e.g. 100k)}"
: "${EPOCH:?set EPOCH (e.g. 60)}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
RUN=./runs_ours/train_${SCALE}

$PY src/eval.py \
  hydra.run.dir=$RUN/eval_sweep_ep${EPOCH} \
  ckpt_path=$RUN/train/checkpoints/epoch_${EPOCH}.pt \
  data/train_datasets=[oursynth_${SCALE}] \
  data/val_datasets=[oursynth_${SCALE}] \
  data/test_datasets=[oursynth_${SCALE}_valsplit] \
  trainer=gpu +trainer.limit_test_batches=0.08 seed=0 callbacks=default \
  evaluation.data_to_extract=['preds','metrics'] \
  evaluation.metrics_to_compute=['multi_match','scan_match_no_dur','kld'] \
  evaluation.eval_root_path=$RUN/eval_sweep_ep${EPOCH} \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "EPOCH_SWEEP_${SCALE}_ep${EPOCH}_DONE"
