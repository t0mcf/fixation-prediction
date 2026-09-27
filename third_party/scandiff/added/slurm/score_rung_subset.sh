#!/bin/bash
#SBATCH --job-name=sd_subscore
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet

# Score one v2 ScanDiff checkpoint on the frozen 400-image subset:
# generate -> convert -> MultiMatch/ScanMatch KLD vs the 16-path AV reference.
# Doubles as a displacement check (spatial mean is printed by the generator).
#
#   sbatch --export=ALL,RUN=oursynth_v2_50k_seed0,CKPT=epoch_60.pt slurm/score_rung_subset.sh
#   RESPACING=250 (default, trajectory scoring) | 1000 (final numbers)
#   CSV defaults to data/eval/v2_scoring_subset/rung_scores.csv (appended)

#   SUBSET=v2_scoring_subset_test (final evaluation only): generate+score on the
#   test-side twin (hydra dataset oursynth_v2_sub400_test), only for the
#   checkpoint selected on the validation subset.

set -euo pipefail
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
: "${RUN:?set RUN, e.g. oursynth_v2_50k_seed0}"
: "${CKPT:=last.pt}"
: "${RESPACING:=250}"
: "${BN_MODE:=eval}"     # eval | recalib (diagnostic, 2026-09-04)
: "${GEN_SEED:=}"        # empty = unseeded (historical); k -> --gen-seed k, tag suffix _s<k>
: "${SUBSET:=v2_scoring_subset}"
case "$SUBSET" in v2_scoring_subset) DSNAME=oursynth_v2_sub400 ;; v2_scoring_subset_test) DSNAME=oursynth_v2_sub400_test ;;
  *) echo "bad SUBSET=$SUBSET" >&2; exit 2 ;; esac
: "${CSV:=$SD/data/eval/$SUBSET/rung_scores.csv}"
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd "$SD"
test -f "$SD/data/eval/$SUBSET/subset_gt.json" || { echo "subset missing: $SUBSET (make_v2_scoring_subset.py)" >&2; exit 4; }

TAG="${RUN}_${CKPT%.pt}_r${RESPACING}"
[ "$BN_MODE" != "eval" ] && TAG="${TAG}_bn${BN_MODE}"
[ -n "$GEN_SEED" ] && TAG="${TAG}_s${GEN_SEED}"
SEED_ARG=""; [ -n "$GEN_SEED" ] && SEED_ARG="--gen-seed $GEN_SEED"
OUT=$SD/data/eval/$SUBSET/gen/$TAG
mkdir -p "$OUT"

"$PY" scripts/generate_v2_subset.py \
  --run-dir "runs_thesis_v2/$RUN" --ckpt "$CKPT" \
  --respacing "$RESPACING" --out "$OUT" --subset-name "$DSNAME" --bn-mode "$BN_MODE" $SEED_ARG

"$PY" tools/scandiff_gen_to_npy.py \
  --gen-pkl "$OUT/generations_${RUN}_${CKPT%.pt}.pkl" \
  --out "$OUT/gen.npy"

"$PY" tools/score_scanpaths.py \
  --human "$SD/data/eval/$SUBSET/subset_gt.json" \
  --generated "$OUT/gen.npy" \
  --label "$TAG" \
  --csv "$CSV"

echo "done: $TAG -> $CSV"
