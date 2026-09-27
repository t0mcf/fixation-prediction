# ScanDiff: our changes and additions

ScanDiff is the work of Cartella et al. (https://github.com/aimagelab/ScanDiff).
All ScanDiff experiments in the thesis ran in a clone of that repository at
upstream commit `9eff26c`. This folder contains everything we changed or
added there, so that the clone can be reconstructed:

```bash
git clone https://github.com/aimagelab/ScanDiff.git scandiff
cd scandiff && git checkout 9eff26c
patch -p1 < <this repo>/third_party/scandiff/upstream.patch
cp -r <this repo>/third_party/scandiff/added/. .
```

The environment is `envs/environment_scandiff.yml` in this repository.

## `upstream.patch`

Changes to seven upstream files (verified: applying the patch to `9eff26c`
reproduces the files used in the experiments byte for byte). The sampling
path is untouched.

| File | Change |
|---|---|
| `configs/train.yaml`, `src/train.py` | `init_from`: loads model weights only (fresh optimizer, epoch 0) to initialize human-gaze training from a synthetic checkpoint; explicit `seed_everything` |
| `configs/trainer/default.yaml`, `src/trainer.py` | `freeze_mode_after_epoch0`: reproduces the authors' effective training mode (batch-normalization statistics frozen and dropout off after the first epoch); `false` keeps the original behaviour |
| `src/callbacks/model_checkpoint.py` | checkpoints additionally store the Python/NumPy/Torch RNG states; `weights_only` loading for `init_from`; RNG states moved to CPU before restoring (resuming failed otherwise) |
| `src/data/components/mit1003/mit1003.py` | feature loading from chunked or memory-mapped files; dataset name taken from the configuration so that the same class serves the synthetic data |
| `src/data/unified_datamodule.py` | persistent data-loader workers |

## `added/`

| Path | Purpose |
|---|---|
| `tools/score_scanpaths.py` | whole-scanpath scoring used for all three models (MM-KLD over the four duration-free MultiMatch components, SM-KLD with the width/height-corrected grid) |
| `tools/eval_reseed.py` | re-seeds generation after the checkpoint is loaded; needed for the second sampling draw, because loading a checkpoint restores its RNG state |
| `tools/rescore_ft_generations_xfam.sh`, `tools/scandiff_gen_to_npy.py` | converts ScanDiff generations and scores them on the MIT1003 cross-model axis |
| `tools/extract_mit1003_human_initial.py` | human reference scanpaths of MIT1003 including the initial fixation |
| `tools/make_v2_scoring_subset.py`, `tools/make_v2_scoring_fulltest.py` | fixed 400-image test subset and full 10,000-image test set for in-domain whole-scanpath scoring |
| `tools/extract_dinov2_features*.py`, `tools/build_scandiff_thesis_v2.py`, `tools/install_scandiff_v2_configs.py` | features and dataset files for synthetic training |
| `scripts/generate_v2_subset.py`, `scripts/generate_pretrain_scandiff.py` | in-domain generation; ScanDiff-generated pretraining corpus (appendix source comparison) |
| `slurm/train_thesis_v2.sh`, `slurm/resume_thesis_v2.sh`, `slurm/score_rung_subset.sh` | synthetic training and checkpoint scoring |
| `slurm/ft_scandiff_mit.sh`, `slurm/ft_scandiff_joint_armB.sh`, `slurm/train_r1_official_recipe*.sh` | MIT1003-only and joint MIT1003 + COCO-FreeView training |
| `slurm/score_ft_valsplit.sh` | validation and test generation for checkpoint selection and final evaluation |
| `paper_reproduction/joint_av1k_followup_20260923/` | joint training initialized from the 1,000-image synthetic checkpoint, with its frozen protocol |
| `configs/data/…` | dataset configurations for the synthetic data and the fixed MIT1003 split |

Other scripts in `added/` are earlier experiments that did not enter the
thesis; they are kept so that the clone can be reconstructed completely.
Some comments in the copies were edited; the code is identical to the version
used for the thesis.
