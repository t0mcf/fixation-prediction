#!/bin/bash
#SBATCH --job-name=sd_keepall
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=inet

# Checkpoint selection for the ScanDiff fine-tuning arms: authors' full
# sampling evaluation on the MIT1003 VALIDATION split (never test — that is
# test-day only). One checkpoint per job. best-val primary per the 2026-08-24
# amendment; the trajectory doubles as the overfitting-shape check (ours
# peaked at epoch 2-3; watch whether ScanDiff's peak is sharper than the
# 5/10-epoch checkpoint grid).
#
#   sbatch --export=ALL,RUN=runs_thesis_v2/ft_mit_scratch_seed0,CKPT=epoch_10.pt slurm/score_ft_valsplit.sh

#   SPLIT=test (final evaluation only): scores the authors' MIT1003 TEST split into
#   data/eval/ft_testscore/ — only for the checkpoint selected on validation.

set -euo pipefail
: "${RUN:?relative run dir, e.g. runs_thesis_v2/ft_mit_scratch_seed0}"
: "${CKPT:=last.pt}"
: "${SPLIT:=valid}"
case "$SPLIT" in valid) SUFFIX=val; SCOREDIR=ft_valscore_keepall ;; test) SUFFIX=test; SCOREDIR=ft_testscore_keepall ;;
  *) echo "bad SPLIT=$SPLIT (valid|test)" >&2; exit 2 ;; esac
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
DATA=$SD/paper_reproduction/data
CKPT_PATH=$SD/$RUN/train/checkpoints/$CKPT
TAG=$(basename "$RUN")_${CKPT%.pt}_${SUFFIX}
OUT=$SD/data/eval/$SCOREDIR/$TAG
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd "$SD"
export PYTHONPATH=$SD:${PYTHONPATH:-}   # tools/ scripts import src.* (fix 2026-09-18)
test -f "$CKPT_PATH" || { echo "missing: $CKPT_PATH" >&2; exit 4; }
mkdir -p "$OUT"

"$PY" tools/eval_keep_all_positions.py --config-path "$SD/configs" \
  hydra.run.dir="$OUT/hydra" \
  ckpt_path="$CKPT_PATH" \
  data/train_datasets="[mit1003]" \
  data/val_datasets="[mit1003]" \
  data/test_datasets="[mit1003]" \
  data.train_datasets.mit1003.root_path="$DATA" \
  data.val_datasets.mit1003.root_path="$DATA" \
  data.test_datasets.mit1003.root_path="$DATA" \
  data.test_datasets.mit1003.split=$SPLIT \
  trainer=gpu seed=0 callbacks=default \
  evaluation.data_to_extract="['preds','metrics']" \
  evaluation.metrics_to_compute="['multi_match','scan_match','scan_match_no_dur','kld']" \
  evaluation.eval_root_path="$OUT/results" \
  diffusion_class=spaced_diffusion data.num_workers=8
echo "done $TAG RUN=$RUN CKPT=$CKPT SPLIT=$SPLIT"
# length-matched cut + scoring on the cross-family axis (control row, label *_lenmatch)
XF=$SD/data/eval/mit_xfam_${SPLIT/valid/validation}
LABEL="scandiff_ft_$(basename "$RUN")_${CKPT%.pt}_lenmatch"; mkdir -p "$XF/gen/$LABEL"
"$PY" tools/scandiff_gen_to_npy_lenmatch.py --gen-pkl "$OUT/results/generations_epoch_*/generations_MIT1003Dataset_test.pkl" \
  --spec "$SD/data/eval/mit1003_human_initial_${SPLIT/valid/validation}.json" --out "$XF/gen/$LABEL/gen.npy"
"$PY" tools/score_scanpaths.py --human "$SD/data/eval/mit1003_human_initial_${SPLIT/valid/validation}.json" \
  --generated "$XF/gen/$LABEL/gen.npy" --label "$LABEL" --csv "$XF/scores.csv" 2>/dev/null | grep "=>"
echo "done lenmatch $LABEL"
