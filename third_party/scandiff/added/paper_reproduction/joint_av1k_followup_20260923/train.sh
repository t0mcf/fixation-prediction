#!/bin/bash
#SBATCH --job-name=sd_ft_joint
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --constraint=inet

# ScanDiff Arm B: the R1 recipe (authors' joint COCO+MIT training, evaluator
# fires once at epoch 0 -> frozen regime, 200 epochs) with ONE change:
# init_from the validation-selected AV-pretrained 1k checkpoint. R1 itself is the from-scratch
# control; the released checkpoint is the external reference. Three-way on
# ScanDiff's own benchmark: released vs scratch vs scratch+AV-pretraining.
#
#   sbatch --export=ALL,SEED=0 slurm/ft_scandiff_joint_armB.sh

set -euo pipefail
: "${SEED:=0}"
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/paper_reproduction/data
: "${INIT_CKPT:=epoch_100.pt}"; SUF=""; [ "$INIT_CKPT" != "last.pt" ] && SUF="_bvinit"
RUN=$SD/paper_reproduction/armB_joint_av1k_seed${SEED}${SUF}
INIT=$SD/runs_thesis_v2/oursynth_v2_1k_seed0/train/checkpoints/$INIT_CKPT
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
export WANDB_MODE=offline
cd "$SD"
"$PY" -B "/mnt/vast-nhr/projects/nim00018/tom/scandiff/paper_reproduction/joint_av1k_followup_20260923/finalize.py" --check
test -f "$INIT" || { echo "missing init: $INIT" >&2; exit 4; }
if [ -e "$RUN/train/checkpoints/last.pt" ]; then
  echo "refusing existing run: $RUN" >&2; exit 3
fi

mkdir -p "$RUN/code_snapshot" "$SD/paper_reproduction/logs"
cp -r src configs jobs "$RUN/code_snapshot/"

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
  logger.wandb.project=scandiff-thesis-v2 logger.wandb.name="armB_joint_av1k_seed${SEED}${SUF}" \
  tags="[armB,joint,avinit,seed${SEED}]" \
  trainer.validation_every_n_epochs=1000 \
  trainer.test_every_n_epochs=1000 \
  trainer.max_epochs=200 \
  train=true test=true \
  evaluation.data_to_extract="['preds','metrics','qualitatives']" \
  evaluation.metrics_to_compute="['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time']" \
  diffusion_class=spaced_diffusion \
  callbacks.model_checkpoint.every_n_epochs=10 \
  ckpt_path=null init_from="$INIT" seed="$SEED"
echo "end=$(date)"
