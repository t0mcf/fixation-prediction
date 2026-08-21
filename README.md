# fixation-prediction

Goal: train a DINOv2-based scanpath-conditioned model for next-fixation prediction that scales to large amounts of synthetic scanpath data.

## Data

Synthetic scanpaths from a foveation model, stored as parquet. Each scanpath has exactly 16 fixations. The first fixation is always the image center (matching standard eye-tracking experiment protocol). The dataset contains ~20.5M scanpaths across ~1.28M ImageNet images (16 per image).

Splits are fixed image lists: 5000 validation images, 10 000 test images, and nested train sets from 1k up to 800k images. They reuse the split DeepGaze III was trained on, so the two models stay comparable without retraining it.

## Model architecture

- **Visual encoder**: frozen DINOv2 ViT-B/14 → 256 spatial patch tokens (768-dim) → linear adapter → d_model
- **Scanpath encoder**: causal transformer (linear coord projection + learned positional embeddings) → one token per past fixation
- **Cross-attention fusion**: image patch tokens (queries) attend to scanpath tokens (keys/values), conditioned on history
- **Decoder**: convolutional upsampler 16×16 → 64×64 → log-probability heatmap

## Training formulation (predict-all)

For each scanpath of length N, a single forward pass produces N−1 heatmaps simultaneously — one for each step t, predicting fixation t+1 given fixations 0..t as context. The causal mask in the scanpath encoder ensures token t only attends to fixations 0..t, preventing information leakage across steps. Two objectives are available via `--loss`: KL against a Gaussian target heatmap, or the log-likelihood at the true fixation cell. The scaling runs use LL.

This is equivalent in spirit to DeepGaze III's maximum-likelihood formulation, but conditions on the full scanpath history instead of a fixed 4-fixation window.

## Spatial baselines

Evaluated on the full val set (96 000 valid prediction steps, σ=8.0, heatmap 64×64).

| method | steps | KL ↓ | LL ↓ | NSS ↑ | AUC ↑ |
|---|---:|---:|---:|---:|---:|
| uniform | 96 000 | 1.4730 | −8.3178 | 0.0000 | 0.5000 |
| center gaussian (σ=12) | 96 000 | 1.4580 | −8.0622 | 0.7556 | 0.7213 |
| empirical density | 96 000 | 1.3244 | −8.0120 | 0.7448 | 0.7138 |

## Branches

- `master`: single random-t prediction per scanpath per epoch (original approach)
- `predict-all`: full predict-all training (this branch, recommended)

## Layout

- `src/` — model, data loading, training loop
- `scripts/` — evaluation, aggregation and figures
- `slurm/` — job scripts currently in use; `slurm/archive/` holds superseded ones
