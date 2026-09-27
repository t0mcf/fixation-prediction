# DeepGaze III: training, sampling and evaluation code

DeepGaze III is the work of Kümmerer, Bethge and Wallis (2022); the model
code is the authors' `deepgaze_pytorch` package
(https://github.com/matthias-k/DeepGaze), used unchanged.

The DeepGaze III experiments of the thesis ran in `scanpather`, a working
repository of V. Hassler (upstream commit `d5ea5a2`) in which the training
pipeline around `deepgaze_pytorch` lives. This folder contains the files from
that repository on which the reported DeepGaze III results depend, as they
were used for the thesis. It is a copy for documentation and reproduction,
not a fork. Some comments were edited; the code is identical to the version
used for the thesis.

The environment is `envs/environment_praktikum.yml` in this repository.

| Path | Purpose |
|---|---|
| `scanpath/train_deepgaze.py`, `scanpath/deepgaze_engine.py` | training entry point (`training.mode` = `imagenet` for synthetic training, `salicon` for SALICON pretraining, `mit1003` for the human-gaze phases) and the three readout networks |
| `scanpath/configs/imagenet.yaml` | synthetic two-stage training; flat (uniform) center bias |
| `scanpath/configs/mit1003*.yaml`, `scanpath/configs/salicon*.yaml` | human-gaze phases with the batch-size-scaled learning rates, SALICON pretraining |
| `scanpath/data/parquet_to_pysaliency.py` | converts the active-vision scanpaths to pysaliency format (out-of-image coordinates clamped to the border) |
| `scanpath/eval_imagenet_fullval.py` | in-domain scores of synthetic checkpoints on the full validation set (the best-validation-LL checkpoint itself is saved by the training loop as `best.pth`) |
| `scanpath/eval_mit1003_fair_dg3.py` | MIT1003 scoring through the shared evaluator of this repository (`src/eval/fair_metrics.py`) |
| `scanpath/sample_scanpaths_dg3_v2.py`, `scanpath/sample_scanpaths_dg3_ft.py` | whole-scanpath generation on the synthetic test subset and on MIT1003 |
| `scanpath/pysaliency_sanity_dg3.py`, `scanpath/pysaliency_eval_ours.py` | cross-check of the shared evaluator against pysaliency's reference evaluation |
| `scripts/tom_runs/dg3_{1k,10k,50k,100k,200k}_5sp*.sh` | synthetic training ladder (five scanpaths per image; `_rep1` = second run; `_resume` = continuation legs) |
| `scripts/tom_runs/train_salicon_tom.sh` | SALICON pretraining for the SALICON reference arm |
| `scripts/tom_runs/dg3_mit1003_fixedsplit.sh` | human-gaze training on the fixed MIT1003 split (702 / 150 / 151) |
| `scripts/tom_runs/score_dg3_fixedsplit_fair.sh`, `sample_dg3_ft_mit.sh`, `sample_dg3_v2_subset.sh` | MIT1003 scoring and whole-scanpath sampling |

Scripts in `scripts/tom_runs/` that are not listed (earlier fold-based
cross-validation, 10-scanpath variants, smoke tests) did not enter the
thesis.

The in-domain conditional test scores of DeepGaze III were computed with the
shared evaluator of this repository (`scripts/eval_synthetic_fair.py`), not
with code from this folder.
