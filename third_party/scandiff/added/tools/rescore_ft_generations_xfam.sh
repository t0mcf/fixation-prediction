#!/bin/bash
# Rescore ScanDiff's saved MIT1003 generations (validation-selected checkpoint
# per fine-tune run, from data/eval/ft_{val,test}score) with tools/score_scanpaths.py
# against mit1003_human_initial_<split>.json — the same code path and reference
# as the ours/DG3 cross-family rows in data/eval/mit_xfam_<split>/scores.csv.
#   bash tools/rescore_ft_generations_xfam.sh validation | test
set -eo pipefail
SPLIT=${1:?validation|test}; case $SPLIT in validation) SC=ft_valscore; SUF=val;; test) SC=ft_testscore; SUF=test;; *) exit 2;; esac
SD=/mnt/vast-nhr/projects/nim00018/tom/scandiff; PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
R=/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/fixation-prediction
SPEC=$SD/data/eval/mit1003_human_initial_${SPLIT}.json; test -f "$SPEC"
cd $SD
tail -n +2 $R/docs/report_tables/scandiff_ft_v2.csv | while IFS=, read human arm init n seed ckpt sm mm; do
  suf=""; [ "$init" = selected ] && suf=_bvinit   # init column added 2026-09-16; selected-init runs carry _bvinit
  case "$human,$arm" in
    mit_only,scratch) run=ft_mit_scratch_seed$seed ;;
    mit_only,av_pretrained) case $n in 1000) sz=1k;; 10000) sz=10k;; 50000) sz=50k;; 100000) sz=100k;; 200000) sz=200k;; esac; run=ft_mit_av${sz}_seed$seed$suf ;;
    joint,scratch) run=r1_official_recipe_seed$seed ;;
    joint,av_pretrained) run=armB_joint_avinit_seed$seed$suf ;;
  esac
  [ "$init" = last ] && continue          # discarded arm (last-epoch init, 2026-09-19)
  for gs in "" "_s1"; do                   # generation seed 0 and the second draw (GEN_SEED=1)
    d=$SD/data/eval/$SC/${run}_${ckpt}_${SUF}${gs}
    pkl=$(ls $d/results/generations_epoch_*/generations_MIT1003Dataset_test.pkl 2>/dev/null | head -1 || true)
    if [ -z "$pkl" ]; then [ -z "$gs" ] && echo "MISSING pickle for $run $ckpt ($d)"; continue; fi
    label=scandiff_ft_${run}_${ckpt}${gs}
    grep -q "^${label}," $SD/data/eval/mit_xfam_${SPLIT}/scores.csv 2>/dev/null && { echo "have $label"; continue; }
    out=$SD/data/eval/mit_xfam_${SPLIT}/gen/$label; mkdir -p $out
    $PY tools/scandiff_gen_to_npy.py --gen-pkl "$pkl" --out $out/gen.npy >/dev/null
    $PY tools/score_scanpaths.py --human $SPEC --generated $out/gen.npy --label $label --csv $SD/data/eval/mit_xfam_${SPLIT}/scores.csv 2>/dev/null | grep -m1 "=>" || true
  done
done
echo RESCORE_DONE
