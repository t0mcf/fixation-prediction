#!/bin/bash
#SBATCH --job-name=sd_conv_check
#SBATCH --output=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/vast-nhr/projects/nim00018/tom/scandiff/slurm/logs/%x_%j.err
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=inet

# Does 50k need 120 epochs? Generate+score m100 at several intermediate
# checkpoints of the (fully-converged) 50k run to see where the realism
# metric actually plateaus.
set -e
export HF_HOME=/mnt/vast-nhr/projects/nim00018/tom/.hf_cache
cd /mnt/vast-nhr/projects/nim00018/tom/scandiff
PY=/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python
for EP in 30 60 90 115; do
  RUN=./runs_ours/train_50k
  $PY src/eval.py \
    hydra.run.dir=$RUN/eval_conv_ep${EP}_hydra \
    ckpt_path=$RUN/train/checkpoints/epoch_${EP}.pt \
    data/train_datasets=[oursynth_50k] \
    data/val_datasets=[mit1003_100] \
    data/test_datasets=[mit1003_100] \
    trainer=gpu seed=0 callbacks=default \
    evaluation.data_to_extract=['preds','metrics'] \
    evaluation.metrics_to_compute=['multi_match','scan_match_no_dur'] \
    evaluation.eval_root_path=$RUN/eval_conv_ep${EP} \
    diffusion_class=spaced_diffusion data.num_workers=8
  echo "EP${EP} GEN DONE"
done
echo "CONVERGENCE CHECK DONE"
