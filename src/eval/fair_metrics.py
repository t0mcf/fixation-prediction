"""
Shared, model-agnostic metric harness for a fair MIT1003 comparison against
DeepGaze III.

Everything here consumes log-probability maps (log-densities) on a fixed
grid plus per-fixation targets, so it can score any model that can emit a
per-step spatial log-density -- ours today, a wrapped DG3 later -- under
identical conventions:

  * LL over uniform   : DG3's exact formula
                        (mean log_density_at_fixation + log(H*W)) / log(2)
                        reported both fixation-weighted and image-averaged
                        (DG3 uses average='image').
  * IG over centerbias: model_LL_bits - centerbias_LL_bits, i.e. bits of
                        information gained over an empirical MIT1003 center
                        bias -- DeepGaze's own headline metric. This removes
                        the free ~1 bit DG3 gets from its bolt-on center-bias
                        prior, so the readout architectures are compared on
                        equal footing.
  * resolution        : predictions and the centerbias are scored on the same
                        grid; resize_logdensity() pools probability mass so
                        the log(H*W) term is identical on both sides.

The center bias is fit as a smoothed 2D histogram of the *training*-fold
fixations (conceptually the same object pysaliency's BaselineModel builds via
KDE). Fitting on the train fold only keeps it leak-free w.r.t. the val fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# coordinate + resolution helpers
# ---------------------------------------------------------------------------

def coords_to_cells(xy: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised [-1, 1] (x, y) -> (row, col) integer cell indices on grid x grid."""
    x, y = xy[:, 0], xy[:, 1]
    col = ((x + 1.0) / 2.0 * (grid - 1)).round().long().clamp(0, grid - 1)
    row = ((y + 1.0) / 2.0 * (grid - 1)).round().long().clamp(0, grid - 1)
    return row, col


def resize_logdensity(log_density: torch.Tensor, target: int) -> torch.Tensor:
    """
    Resize log-density maps to target x target by pooling probability mass:
    exp -> bilinear interp -> renormalise -> log. Keeps sum(prob) == 1.

    input : (N, H, W) log-softmax
    output: (N, target, target) log-softmax
    """
    probs = log_density.exp().unsqueeze(1)                       # (N,1,H,W)
    probs = F.interpolate(probs, size=(target, target),
                          mode="bilinear", align_corners=False).squeeze(1)
    probs = probs.clamp_min(0)
    probs = probs / probs.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(probs.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# empirical center bias
# ---------------------------------------------------------------------------

def fit_centerbias(
    train_xy: np.ndarray,
    grid: int,
    smooth_frac: float = 0.05,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Empirical MIT1003 center bias: smoothed 2D histogram of training fixation
    positions, normalised to a log-density on grid x grid.

    train_xy   : (M, 2) normalised [-1, 1] fixation coords (targets in train fold)
    smooth_frac: Gaussian sigma as a fraction of grid width (0.05 -> ~3 cells @64)

    returns    : (grid, grid) log-density, logsumexp == 0
    """
    xy = torch.from_numpy(np.asarray(train_xy, dtype=np.float32))
    row, col = coords_to_cells(xy, grid)
    hist = torch.zeros(grid, grid, dtype=torch.float32)
    hist.index_put_((row, col), torch.ones(len(row)), accumulate=True)

    # Gaussian smoothing via separable conv
    sigma = max(smooth_frac * grid, 0.5)
    radius = max(int(3 * sigma), 1)
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    k1d = k1d / k1d.sum()
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    h = hist.view(1, 1, grid, grid)
    h = F.conv2d(h, kx, padding=(0, radius))
    h = F.conv2d(h, ky, padding=(radius, 0)).view(grid, grid)

    h = h + h.mean() * 1e-3           # floor so no cell is -inf
    prob = h / h.sum()
    return torch.log(prob.clamp_min(1e-12)).to(device)


# ---------------------------------------------------------------------------
# tuned center bias (DG3-parity): cross-validated bandwidth + eps mixture
# ---------------------------------------------------------------------------

# candidate grids bracket DG3's MIT1003-tuned values (bandwidth 0.0215 of image
# size; eps-mixture with uniform). eps is resolution-dependent, so the grid is
# wide; 0.0 is excluded because empty far-field cells would give -inf LL.
CB_BANDWIDTHS = (0.01, 0.015, 0.0215, 0.03, 0.04, 0.05, 0.07)
CB_EPSILONS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def _smooth_hist(hist: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian smoothing of a (..., grid, grid) histogram stack."""
    grid = hist.shape[-1]
    radius = max(int(3 * sigma), 1)
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    k1d = k1d / k1d.sum()
    h = hist.reshape(-1, 1, grid, grid)
    h = F.conv2d(h, k1d.view(1, 1, 1, -1), padding=(0, radius))
    h = F.conv2d(h, k1d.view(1, 1, -1, 1), padding=(radius, 0))
    return h.reshape(hist.shape)


def fit_centerbias_tuned(
    per_image_xy: list[np.ndarray],
    grid: int,
    bandwidths: tuple[float, ...] = CB_BANDWIDTHS,
    epsilons: tuple[float, ...] = CB_EPSILONS,
    n_chunks: int = 10,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    DG3-parity center bias: smoothed histogram whose bandwidth and uniform-
    mixture eps are selected by image-wise cross-validation *within* the given
    fixations -- the same tuning opportunity DeepGaze's pysaliency
    BaselineModel got (their MIT1003 values: bandwidth 0.0215, tuned eps).

    per_image_xy: one (n_i, 2) array of normalised [-1,1] fixation coords per
                  image (ALL fixations of that image, first included).
    Selection: images are split into n_chunks; for each candidate
    (bandwidth, eps) the density is fit on the other chunks (linearity of
    smoothing: fit = smooth(total) - smooth(chunk)) and scored as mean
    log-density of the held-out chunk's fixations. Winner is refit on all
    fixations. Model: p = (1-eps) * smoothed_hist_norm + eps * uniform.

    returns: (grid, grid) log-density, logsumexp == 0
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(per_image_xy))
    chunk_of = np.empty(len(per_image_xy), dtype=int)
    for pos, img in enumerate(order):
        chunk_of[img] = pos % n_chunks

    # per-chunk raw histograms + held-out fixation cells
    chunk_hist = torch.zeros(n_chunks, grid, grid)
    chunk_cells: list[list[torch.Tensor]] = [[] for _ in range(n_chunks)]
    for img, xy in enumerate(per_image_xy):
        if len(xy) == 0:
            continue
        t = torch.from_numpy(np.asarray(xy, dtype=np.float32))
        row, col = coords_to_cells(t, grid)
        c = chunk_of[img]
        chunk_hist[c].index_put_((row, col), torch.ones(len(row)), accumulate=True)
        chunk_cells[c].append(row * grid + col)
    flat_cells = [torch.cat(cs) if cs else torch.zeros(0, dtype=torch.long)
                  for cs in chunk_cells]
    total_hist = chunk_hist.sum(dim=0)
    uniform = 1.0 / (grid * grid)

    best = (-math.inf, bandwidths[0], epsilons[0])
    for bw in bandwidths:
        sigma = max(bw * grid, 0.5)
        sm_total = _smooth_hist(total_hist, sigma)
        sm_chunks = _smooth_hist(chunk_hist, sigma)
        for eps in epsilons:
            ll_sum, n_fix = 0.0, 0
            for c in range(n_chunks):
                cells = flat_cells[c]
                if len(cells) == 0:
                    continue
                fit = (sm_total - sm_chunks[c]).clamp_min(0)
                mass = fit.sum()
                if mass <= 0:
                    continue
                prob = (1.0 - eps) * (fit / mass) + eps * uniform
                ll_sum += float(torch.log(prob.reshape(-1)[cells]).sum())
                n_fix += len(cells)
            ll = ll_sum / max(n_fix, 1)
            if ll > best[0]:
                best = (ll, bw, eps)

    ll, bw, eps = best
    if verbose:
        print(f"    center bias tuned: bandwidth={bw} eps={eps:g} "
              f"(held-out LL {ll / math.log(2):.4f} bits/fix)", flush=True)

    sigma = max(bw * grid, 0.5)
    fit = _smooth_hist(total_hist, sigma).clamp_min(0)
    prob = (1.0 - eps) * (fit / fit.sum()) + eps * uniform
    logp = torch.log(prob.clamp_min(1e-30))
    logp = logp - torch.logsumexp(logp.reshape(-1), dim=0)
    return logp.to(device)


# ---------------------------------------------------------------------------
# metric accumulation
# ---------------------------------------------------------------------------

@dataclass
class FairMetricAccumulator:
    """
    Accumulate per-fixation log-density values, grouped by image, so LL can be
    reported both fixation-weighted and image-averaged (DG3 style), and as
    information gain over a supplied center bias.

    All densities are scored at `grid` resolution; the caller is responsible
    for passing log_density and centerbias already at that resolution.
    """
    grid: int
    centerbias: torch.Tensor  # (grid, grid) log-density, or None

    def __post_init__(self):
        self._log_norm = math.log(self.grid * self.grid)
        # per-image running sums: img_idx -> [sum_model_log, sum_cb_log, count]
        self._by_image: dict[int, list[float]] = {}

    def add(self, log_density: torch.Tensor, target_xy: torch.Tensor,
            img_idx: torch.Tensor) -> None:
        """
        log_density: (N, grid, grid) log-softmax predictions
        target_xy  : (N, 2) normalised [-1, 1] targets
        img_idx    : (N,) image index per sample (for image-averaging)
        """
        row, col = coords_to_cells(target_xy, self.grid)
        idx = torch.arange(len(row))
        model_log = log_density[idx, row, col]                      # (N,)
        if self.centerbias is not None:
            cb_log = self.centerbias[row, col]                      # (N,)
        else:
            cb_log = torch.full_like(model_log, -self._log_norm)    # uniform

        for i in range(len(row)):
            im = int(img_idx[i])
            acc = self._by_image.setdefault(im, [0.0, 0.0, 0.0])
            acc[0] += float(model_log[i])
            acc[1] += float(cb_log[i])
            acc[2] += 1.0

    def result(self) -> dict[str, float]:
        if not self._by_image:
            return {}
        ln2 = math.log(2.0)
        # fixation-weighted: pool all fixations equally
        tot_model = sum(v[0] for v in self._by_image.values())
        tot_cb    = sum(v[1] for v in self._by_image.values())
        tot_n     = sum(v[2] for v in self._by_image.values())
        ll_uniform_fix = (tot_model / tot_n + self._log_norm) / ln2
        cb_uniform_fix = (tot_cb    / tot_n + self._log_norm) / ln2

        # image-averaged: per-image mean log first, then mean over images
        per_img_model = [v[0] / v[2] for v in self._by_image.values()]
        per_img_cb    = [v[1] / v[2] for v in self._by_image.values()]
        m_model = float(np.mean(per_img_model))
        m_cb    = float(np.mean(per_img_cb))
        ll_uniform_img = (m_model + self._log_norm) / ln2
        cb_uniform_img = (m_cb    + self._log_norm) / ln2

        return {
            "ll_uniform_fix":  ll_uniform_fix,
            "ll_uniform_img":  ll_uniform_img,
            "ig_centerbias_fix": ll_uniform_fix - cb_uniform_fix,
            "ig_centerbias_img": ll_uniform_img - cb_uniform_img,
            "centerbias_ll_uniform_img": cb_uniform_img,
            "n_fixations": int(tot_n),
            "n_images": len(self._by_image),
        }


# ---------------------------------------------------------------------------
# saccade-delta prior: "how much of the LL gap is trivial saccade geometry,
# not image content" -- a 1st-order Markov baseline, no image seen at all.
# ---------------------------------------------------------------------------

# candidates: saccade-length distributions are usually much more peaked than
# the center bias's spatial spread, so the bandwidth grid skews narrower.
SACC_BANDWIDTHS = (0.005, 0.01, 0.0215, 0.03, 0.05, 0.08, 0.12)
SACC_EPSILONS = CB_EPSILONS


def scanpath_deltas_by_image(per_image_scanpaths: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    per_image_scanpaths: one list of (L_i, 2) arrays PER IMAGE, each array one
    scanpath (fixations in order, normalised [-1,1]) -- scanpath boundaries
    must be preserved so deltas are never taken across different subjects.

    Returns: one (n_i, 2) array of (dx, dy) = fixation[t+1] - fixation[t] per
    image, pooling all consecutive within-scanpath pairs.
    """
    out = []
    for sps in per_image_scanpaths:
        deltas = []
        for sp in sps:
            if len(sp) >= 2:
                deltas.append(sp[1:] - sp[:-1])
        out.append(np.concatenate(deltas, axis=0) if deltas
                   else np.zeros((0, 2), np.float32))
    return out


def fit_saccade_prior_tuned(
    delta_xy_by_image: list[np.ndarray],
    grid: int,
    bandwidths: tuple[float, ...] = SACC_BANDWIDTHS,
    epsilons: tuple[float, ...] = SACC_EPSILONS,
    n_chunks: int = 10,
    seed: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    1st-order Markov saccade-delta prior: p(next_cell - prev_cell), a single
    translation-invariant kernel over integer cell offsets, image-wise
    cross-validated for bandwidth + eps exactly like fit_centerbias_tuned --
    same rigor, same tuning opportunity, just conditioned on displacement
    instead of absolute position. No image content anywhere in this baseline.

    delta_xy_by_image: one (n_i, 2) array of (dx, dy) in [-2,2]-normalised
                       units per image (from scanpath_deltas_by_image).

    returns: (2*grid-1, 2*grid-1) log-density over delta cells, logsumexp==0.
             index (grid-1, grid-1) == delta (0, 0).
    """
    size = 2 * grid - 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(delta_xy_by_image))
    chunk_of = np.empty(len(delta_xy_by_image), dtype=int)
    for pos, img in enumerate(order):
        chunk_of[img] = pos % n_chunks

    def to_delta_cells(dxdy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # dx, dy in [-2, 2] (difference of two [-1,1] coords) -> cell offset
        # in [-(grid-1), grid-1], then shifted to a non-negative index.
        d_row = np.round(dxdy[:, 1] / 2.0 * (grid - 1)).astype(np.int64)
        d_col = np.round(dxdy[:, 0] / 2.0 * (grid - 1)).astype(np.int64)
        d_row = np.clip(d_row, -(grid - 1), grid - 1) + (grid - 1)
        d_col = np.clip(d_col, -(grid - 1), grid - 1) + (grid - 1)
        return d_row, d_col

    chunk_hist = torch.zeros(n_chunks, size, size)
    chunk_cells: list[list[np.ndarray]] = [[] for _ in range(n_chunks)]
    for img, dxdy in enumerate(delta_xy_by_image):
        if len(dxdy) == 0:
            continue
        d_row, d_col = to_delta_cells(np.asarray(dxdy, dtype=np.float32))
        c = chunk_of[img]
        chunk_hist[c].index_put_(
            (torch.from_numpy(d_row), torch.from_numpy(d_col)),
            torch.ones(len(d_row)), accumulate=True)
        chunk_cells[c].append(d_row * size + d_col)
    flat_cells = [torch.from_numpy(np.concatenate(cs)) if cs
                  else torch.zeros(0, dtype=torch.long)
                  for cs in chunk_cells]
    total_hist = chunk_hist.sum(dim=0)
    uniform = 1.0 / (size * size)

    best = (-math.inf, bandwidths[0], epsilons[0])
    for bw in bandwidths:
        sigma = max(bw * grid, 0.5)
        sm_total = _smooth_hist(total_hist, sigma)
        sm_chunks = _smooth_hist(chunk_hist, sigma)
        for eps in epsilons:
            ll_sum, n_delta = 0.0, 0
            for c in range(n_chunks):
                cells = flat_cells[c]
                if len(cells) == 0:
                    continue
                fit = (sm_total - sm_chunks[c]).clamp_min(0)
                mass = fit.sum()
                if mass <= 0:
                    continue
                prob = (1.0 - eps) * (fit / mass) + eps * uniform
                ll_sum += float(torch.log(prob.reshape(-1)[cells]).sum())
                n_delta += len(cells)
            ll = ll_sum / max(n_delta, 1)
            if ll > best[0]:
                best = (ll, bw, eps)

    ll, bw, eps = best
    if verbose:
        print(f"    saccade prior tuned: bandwidth={bw} eps={eps:g} "
              f"(held-out LL {ll / math.log(2):.4f} bits/step)", flush=True)

    sigma = max(bw * grid, 0.5)
    fit = _smooth_hist(total_hist, sigma).clamp_min(0)
    prob = (1.0 - eps) * (fit / fit.sum()) + eps * uniform
    logp = torch.log(prob.clamp_min(1e-30))
    logp = logp - torch.logsumexp(logp.reshape(-1), dim=0)
    return logp


def saccade_prior_logmaps(
    delta_logdensity: torch.Tensor,
    grid: int,
    prev_row: torch.Tensor,
    prev_col: torch.Tensor,
    extra_log_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Place the fitted (2*grid-1, 2*grid-1) saccade-delta kernel at each sample's
    previous-fixation cell, crop to the visible grid, optionally add another
    log-prior in log-space (e.g. the tuned center bias, for the combined
    "position + saccade geometry, still zero image content" baseline), and
    renormalise (log-softmax) -- exactly how our own model's finalizer
    combines the decoder logits with the center-bias prior.

    prev_row, prev_col: (N,) integer cell indices of the previous fixation.
    extra_log_prior: (grid, grid) log-density (e.g. tuned center bias),
                     broadcast-added before renormalisation, or None.

    returns: (N, grid, grid) log-softmax density.
    """
    N = len(prev_row)
    size = delta_logdensity.shape[0]  # == 2*grid-1
    out = torch.empty(N, grid, grid)
    for i in range(N):
        r0, c0 = int(prev_row[i]), int(prev_col[i])
        patch = delta_logdensity[grid - 1 - r0: size - r0,
                                 grid - 1 - c0: size - c0]
        out[i] = patch
    if extra_log_prior is not None:
        out = out + extra_log_prior.unsqueeze(0)
    return F.log_softmax(out.reshape(N, -1), dim=-1).reshape(N, grid, grid)
