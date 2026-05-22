# Engineering Log

Running notes on implementation decisions, profiling findings, and optimizations.
Intended as a reference for the internship report.

---

## 1. Training formulation: predict-all

### Motivation

The initial training approach (branch `master`) sampled a single random prediction step $t$ per scanpath per epoch. For a scanpath of length 16, this means each training step only uses 1 out of 15 possible supervision signals. This is data-inefficient: each image-scanpath pair contributes only one gradient update per epoch regardless of how much supervisory information it contains.

### Approach

We switched to a **predict-all** formulation, also used by DeepGaze III (Kümmerer et al., 2022). A single forward pass produces $T-1 = 15$ heatmaps simultaneously — one for each step $t \in \{0, \ldots, T-2\}$ — predicting fixation $t+1$ given fixations $0 \ldots t$ as context. This is implemented via a causal mask in the scanpath encoder: token $t$ attends only to positions $0 \ldots t$, preventing information leakage across steps.

The loss is KL divergence averaged over all valid (scanpath, step) pairs in the batch:

$$\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{(b,t) \in \mathcal{V}} \text{KL}(g_{b,t} \| \hat{p}_{b,t})$$

where $g_{b,t}$ is a Gaussian heatmap (σ=2px in 64×64 space) centered at the true fixation, $\hat{p}_{b,t}$ is the model's predicted log-probability heatmap, and $\mathcal{V}$ is the set of valid pairs ($t < N_b - 1$ for scanpath $b$ of length $N_b$).

### Effect

Each forward pass now provides up to 15× more gradient signal per scanpath. This is equivalent in data efficiency to the original approach running 15 epochs for every 1 epoch of predict-all. An overfit test on 8 images confirmed convergence: training KL dropped from ~3.5 to ~0.5 in approximately 1680 steps, well below the empirical density baseline of 1.32.

---

## 2. Profiling and performance optimization

### 2.1 Motivation

Before committing to long training runs (tens of thousands of steps), we profiled the model to identify computational bottlenecks. This is important because training on 50k+ images requires efficient use of GPU time.

### 2.2 Methodology

We used the PyTorch profiler (`torch.profiler.profile`) with both CPU and CUDA activity recording enabled. To isolate model computation from data loading overhead, we constructed synthetic batches directly on the GPU (random tensors of the correct shapes) rather than using the real dataloader. This gives a clean view of the model's compute graph without I/O noise.

**Profiling configuration:** batch_size=32, seq_len=16 (T=15 prediction steps), fusion_layers=4, A100 40GB GPU. Three warmup steps were run before recording to ensure CUDA kernels were initialized and JIT-compiled.

### 2.3 Initial findings: the sequential T-loop

The original `ScanpathModel.forward()` iterates over all T=15 prediction steps in a Python loop:

```python
for t in range(T):
    eff_len = torch.clamp(prefix_len, max=t+1)
    fused = image_tokens
    for layer in self.fusion:
        fused = layer(fused, scanpath_tokens, eff_len)
    logit = self.decoder(fused)
    heatmaps.append(F.log_softmax(...))
```

Each loop iteration dispatches a separate sequence of CUDA kernels. With 4 fusion layers and a multi-layer decoder per step, this amounts to approximately 135 separate CUDA kernel launches per training step, each dispatched sequentially from Python.

**The problem:** A GPU executes work efficiently when given large contiguous batches of computation. Sequential Python-level dispatch means the GPU completes each kernel and then sits idle while Python prepares the next call. Profiling confirmed this: total CUDA compute time per step was approximately 782ms, while the actual wall-clock step time was approximately 2 seconds — indicating the GPU was idle for roughly 60% of each step.

Additionally, the backward pass required Python's autograd engine to traverse a computation graph 60 nodes deep (15 steps × 4 fusion layers), resulting in approximately 506ms of CPU-side dispatch overhead per step just for the backward pass.

### 2.4 Fix: vectorizing the T-loop

Instead of running the fusion layers and decoder 15 times sequentially on tensors of shape $(B, N, d)$, we tile the batch dimension to $(B \times T, N, d)$ and run a single batched forward pass:

```python
# build effective prefix length for every (sample, step) pair
t_idx = torch.arange(T, device=image.device)
eff_len_flat = torch.min(
    prefix_len.unsqueeze(1),       # (B, 1)
    (t_idx + 1).unsqueeze(0),      # (1, T)
).reshape(B * T)                   # (B*T,)

# tile inputs and run fusion + decoder once
image_tiled = image_tokens.repeat_interleave(T, dim=0)    # (B*T, N, d)
scan_tiled  = scanpath_tokens.repeat_interleave(T, dim=0) # (B*T, T, d)

fused = image_tiled
for layer in self.fusion:
    fused = layer(fused, scan_tiled, eff_len_flat)

logit = model.decoder(fused)           # (B*T, H, W)
log_h = F.log_softmax(...).view(B, T, H, W)
```

The `eff_len_flat` tensor encodes both the causal constraint and variable sequence lengths: for virtual sample $b \cdot T + t$, it equals $\min(t+1, \text{prefix\_len}_b)$, which is exactly the masking previously applied per-loop-iteration. The `CrossAttentionFusion` layer requires no changes — it is agnostic to batch size.

**The key insight:** the total arithmetic is identical. What changes is that instead of 135 small sequential CUDA kernel launches, we issue approximately 9 large ones. The GPU stays busy for longer per launch and idles far less.

### 2.5 Results of vectorization

Profiling results before and after (batch_size=32, T=15, 4 fusion layers):

| metric | before (loop) | after (vectorized) |
|---|---:|---:|
| Total CUDA time / step | ~782ms | ~649ms |
| `aten::mm` CUDA / step | ~121ms | ~78ms |
| `aten::upsample_bilinear2d` calls | 150 (30/step) | 10 (2/step) |
| Backward CPU dispatch / step | ~506ms | ~16ms |
| Self CPU total (5 steps) | 6.6s | 2.1s |

The backward CPU dispatch overhead dropped by 97% — from 506ms to 16ms per step — because the autograd graph shrank from 60 nodes deep to 4. Total CPU overhead dropped by approximately 3×.

### 2.6 Remaining bottleneck: bilinear upsampling

After vectorization, profiling revealed that `aten::upsample_bilinear2d` became the second largest CUDA consumer at ~95ms/step (14.67% of CUDA time), despite only 2 kernel launches per step. Each call processes a tensor of shape $(B \times T, C, H, W) = (480, 256, 32, 32)$ or $(480, 128, 64, 64)$, totaling approximately 1.5GB of data per step.

This is approximately 100× slower than the theoretical A100 memory bandwidth limit (~1ms expected vs ~95ms observed), indicating that PyTorch's `nn.Upsample(mode='bilinear')` kernel is poorly optimized for this tensor shape. **[TODO: add result after replacing with transposed convolutions]**

### 2.7 Fix: replacing bilinear upsample — transposed conv attempt, then nearest+conv

**First attempt: transposed convolution.** Each `nn.Upsample(mode='bilinear') + nn.Conv2d` pair was replaced with a single `nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)`. Transposed convolutions with `kernel_size=2, stride=2` perform exactly 2× spatial upsampling and are natively supported by cuDNN with well-optimized SGEMM kernels, addressing the bilinear upsample performance issue. Initial profiling confirmed the speedup (see section 2.8).

**Problem: checkerboard artifacts.** `ConvTranspose2d` with `kernel_size=2, stride=2` has no kernel overlap between adjacent output positions — each output pixel is computed from a different, non-overlapping subset of the kernel. This produces a structural periodic pattern (checkerboard) in the output heatmaps. Crucially, this is not a training stability issue that resolves with more steps; it is an inductive bias baked into the architecture. The pattern was visible in wandb heatmap logs and did not diminish over training.

**Final fix: nearest upsample + conv.** Each upsampling stage was replaced with `nn.Upsample(scale_factor=2, mode='nearest') + nn.Conv2d(kernel_size=3, padding=1)`. Nearest upsampling duplicates each pixel without introducing frequency artifacts. The subsequent 3×3 convolution has full overlap across the upsampled grid, so it can learn to produce smooth outputs. Unlike bilinear, nearest upsample is a trivial memory copy (fast); unlike transposed conv, the following conv has no structural periodicity. The conv weights are learned, so the network can adapt the smoothing to the task.

### 2.8 Full optimization results

Three profiling runs were conducted (batch_size=32, T=15, 4 fusion layers, A100 40GB GPU):

| metric | loop + bilinear | vectorized + bilinear | vectorized + transposed conv |
|---|---:|---:|---:|
| CUDA per step | ~242ms | ~250ms | **~152ms** |
| `fusion_decoder_loop` CUDA | ~152ms/step | ~160ms/step | **~61ms/step** |
| `upsample_bilinear2d` | 76ms, 150 calls | 95ms, 10 calls | **eliminated** |
| `ConvolutionBackward` CUDA | ~28ms/step | ~30ms/step | **~9ms/step** |
| backward CPU dispatch | ~506ms/step | ~16ms/step | ~16ms/step |
| Self CPU total (5 steps) | 6.6s | 2.1s | **1.5s** |

The vectorization alone had a negligible effect on total CUDA time (the upsample cost actually increased per call due to larger tensor size), but dramatically reduced CPU overhead: backward dispatch dropped from 506ms to 16ms per step, and total CPU overhead dropped 3×. The transposed convolution then cut CUDA per step by a further 37% by eliminating the inefficient bilinear kernel. Combined, the two optimizations reduced total CUDA time per step by ~37% and CPU overhead by ~10×. The dominant remaining costs are cross-attention and linear layer matrix multiplications (`aten::addmm`, ~110ms/step) and the frozen DINOv2 encoder (`encode_image`, ~91ms/step), both of which are expected and not reducible without architectural changes.

---

### 2.9 DINOv2 encoder deduplication

With the grouped sampler (`paths_per_image=16`, `batch_size=128`), each batch contains 16 scanpaths from each of 8 unique images — but the image encoder was previously called on all 128 image tensors, running DINOv2 redundantly 16× per unique image.

The fix is a one-line change in `ScanpathModel.forward()`: extract the unique images, run the encoder once per unique image, then tile the result:

```python
if paths_per_image > 1 and B % paths_per_image == 0:
    unique_images = image[::paths_per_image]           # (B//P, 3, H, W)
    image_tokens = self.encode_image(unique_images).repeat_interleave(
        paths_per_image, dim=0
    )                                                  # (B, N, d_model)
```

`repeat_interleave` is differentiable, so gradients flow correctly back through the image adapter. Correctness was verified by confirming identical loss values at each step compared to the non-deduplicated run.

### 2.10 Cumulative throughput improvements

All optimizations were measured on A100 40GB with batch_size=128, paths_per_image=16, 50k images:

| optimization stack | steps/hour |
|---|---:|
| original (loop + bilinear upsample, no AMP) | ~1770 |
| + vectorized T-loop + transposed conv (no AMP) | ~3250 |
| + DINOv2 deduplication + AMP | **~7200** |

Total speedup: **4×** over the original implementation.

---

## 3. Scale-up training results

### 3.1 Experimental setup

Three initial scale-up runs were conducted to establish baseline performance and test the scaling hypothesis:

| run | images | σ | epochs | best NSS (epoch) | final NSS |
|---|---:|---:|---:|---:|---:|
| 50k σ=2.0 | 50 000 | 2.0 | 15 | 3.347 (ep 9) | 3.294 |
| 50k σ=4.0 | 50 000 | 4.0 | 15 | 2.875 (ep 9) | 2.834 |
| 200k σ=2.0 | 200 000 | 2.0 | 5 | — | 4.084 (still improving) |

### 3.2 Overfitting on 50k

Both 50k runs show a clear overfitting pattern: validation NSS and AUC peak around epoch 7–9 then degrade, while training KL continues to decrease. The best 50k checkpoint is therefore around epoch 7–9, not the final checkpoint. 15 epochs is too many for 50k images with this model capacity.

### 3.3 σ=2.0 vs σ=4.0

The σ=4.0 run reaches lower KL loss (softer targets are easier to fit) but worse NSS and AUC (2.875 vs 3.347). Softer Gaussian targets smooth out the supervision signal, allowing the model to produce diffuse predictions that are penalised less. σ=2.0 forces the model to be more precise, which corresponds better to what NSS/AUC measure. All subsequent runs use σ=2.0.

### 3.4 Scaling signal

The 200k run reaches NSS 4.08 after just 5 epochs, already +22% over the 50k best (3.35 at epoch 9). The run was still improving monotonically with no sign of plateauing. This is the key result for the thesis: **model quality improves substantially with more training data**, a property DeepGaze III (fixed small readout network) cannot exploit.

Full 15-epoch 200k runs are currently in progress to characterise the scaling curve beyond epoch 5.

---

## 4. Loss function: KL divergence vs. negative log-likelihood

### 4.1 Motivation

The KL divergence loss trains the model to match a Gaussian heatmap $g_\sigma$ centred at the true fixation:

$$\mathcal{L}_\text{KL} = \text{KL}(g_\sigma \| \hat{p})$$

This requires choosing $\sigma$ — a hyperparameter that controls the softness of the supervision. Too large and the model learns to predict diffuse blobs; too small and the loss becomes numerically unstable near the boundary of the heatmap. Moreover, $g_\sigma$ is a smoothed proxy for the ground truth, not the ground truth itself.

DeepGaze III instead maximises the log-likelihood directly at the true fixation location — the probability mass the model assigns to exactly the pixel where the fixation occurred:

$$\mathcal{L}_\text{NLL} = -\log \hat{p}(y_t, x_t)$$

This is equivalent to KL divergence against a Dirac delta at the fixation, and removes $\sigma$ as a hyperparameter entirely. The signal is sharper: the model is rewarded only for placing probability mass at the correct pixel, not for being close in a Gaussian sense.

### 4.2 Implementation

The `multi_step_ll_score` function was already implemented as a monitoring metric. A `--loss {kl, ll}` argument was added to `train.py` to select which objective to backpropagate through. Both KL and LL are always logged regardless of which is used for training, enabling direct comparison.

For the NLL run the training loss is $-\text{LL}$ (negative, since we minimise). Validation metrics (NSS, AUC, KL, LL) are computed identically for both runs so results are directly comparable.

### 4.3 Experiment

A direct comparison run is underway: `train_200k_ll_15ep` vs `train_200k_sigma2_15ep`, identical in all other respects (200k images, 15 epochs, same architecture and optimiser). Results pending.

---

## 5. Visual scanpath features

### 5.1 Motivation

The scanpath encoder currently receives only (x, y) coordinate pairs. Each fixation token encodes *where* the eye was, but not *what was seen* there. This limits what the model can learn about fixation history: two very different visual experiences (fixating a dog's face vs. a blank wall) produce identical scanpath tokens if they occur at the same coordinates.

Human fixation behaviour is driven by semantic content: inhibition of return operates partly on object identity, not just location; people tend to fixate on semantically coherent regions and avoid content they have already processed. A model that cannot distinguish visual content at past fixations cannot learn these patterns.

### 5.2 Approach

After computing `image_tokens` (B, N, d_model) — the DINOv2 patch features projected to d_model space — we look up the feature at each past fixation location:

```python
col = ((prefix[..., 0] + 1) / 2 * (gs - 1)).round().long().clamp(0, gs - 1)
row = ((prefix[..., 1] + 1) / 2 * (gs - 1)).round().long().clamp(0, gs - 1)
patch_idx = row * gs + col  # (B, T)
patch_feats = image_tokens[b_idx, patch_idx]  # (B, T, d_model)
```

These are injected at the **input** of the scanpath encoder, before its causal transformer layers, via a learned linear projection:

```python
x = coord_proj(prefix) + patch_feat_proj(patch_feats) + pos_embed(positions)
```

Injecting at the encoder input — rather than inside the fusion layers — is the key design choice. The causal transformer in the scanpath encoder can then reason over the full *visual sequence*, relating what was seen at each step to what was seen at others. By the time scanpath tokens reach the cross-attention fusion, they already carry semantically enriched history, not just spatial history.

### 5.3 Cost

- Additional parameters: one `Linear(d_model, d_model)` layer = 65,792 parameters (negligible)
- Memory: one additional (B, T, d_model) tensor — ~1MB at batch_size=128, T=15
- Compute: a single gather operation (patch lookup) + one small matmul — negligible against cross-attention and DINOv2 encoder costs
- No additional DINOv2 forward passes; features are looked up from already-computed `image_tokens`

Gradients from the scanpath path now also flow back through `image_adapter`, giving it gradient signal from two sources. This may marginally improve the adapter's learned projection.

### 5.4 Experiment

`train_200k_visfeat_15ep` — identical to `train_200k_sigma2_15ep` except `--visual-scanpath-features`. Results pending.

---

## 6. Pending / upcoming

- [x] Measure throughput (steps/hour) after all optimizations → ~7200 steps/hour
- [x] Scale-up training runs: 50k σ=2.0, 50k σ=4.0, 200k σ=2.0 (5ep)
- [x] Sigma ablation: σ=2.0 clearly better than σ=4.0 on NSS/AUC
- [ ] 200k full runs: `train_200k_sigma2_15ep`, `train_200k_ll_15ep`, `train_200k_visfeat_15ep` — in progress
- [ ] Evaluate best checkpoint against spatial baselines (uniform, center gaussian, empirical density)
- [ ] Analyse scaling curve: 50k → 200k → potentially more data
