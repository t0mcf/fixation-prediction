"""Training losses and heatmap utilities for scanpath prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_gaussian_heatmaps_batch(
    fixations: torch.Tensor,
    size: int,
    sigma: float,
) -> torch.Tensor:
    """
    Vectorised Gaussian heatmap generation for a batch of fixation sequences.

    input:
        fixations: (B, T, 2) in [-1, 1] normalised coords
        size: heatmap side length (e.g. 64)
        sigma: Gaussian sigma in heatmap pixels

    output:
        heatmaps: (B, T, size, size) normalised probability maps
    """
    B, T, _ = fixations.shape
    device = fixations.device

    x = fixations[..., 0]  # (B, T)
    y = fixations[..., 1]  # (B, T)

    px = (x + 1.0) / 2.0 * (size - 1)  # (B, T)
    py = (y + 1.0) / 2.0 * (size - 1)  # (B, T)

    grid = torch.arange(size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")  # (size, size)

    # (B, T, size, size)
    dx = grid_x[None, None] - px[:, :, None, None]
    dy = grid_y[None, None] - py[:, :, None, None]

    heatmaps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    heatmaps = heatmaps / heatmaps.sum(dim=(-1, -2), keepdim=True)

    return heatmaps


def multi_step_kl_loss(
    log_preds: torch.Tensor,
    targets: torch.Tensor,
    fixations_len: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence averaged over all valid (sample, step) pairs.

    input:
        log_preds: (B, T, H, W) log-probability predictions
        targets: (B, T, H, W) Gaussian target heatmaps
        fixations_len: (B,) total fixation count per sample;
                       step t is valid for sample b when t < fixations_len[b] - 1

    output:
        loss: scalar
    """
    B, T, H, W = log_preds.shape
    device = log_preds.device

    valid_mask = (
        torch.arange(T, device=device).unsqueeze(0) < (fixations_len - 1).unsqueeze(1)
    )  # (B, T)

    total_valid = valid_mask.sum()
    if total_valid == 0:
        return (log_preds * 0).sum()

    log_preds_flat = log_preds.view(B, T, -1)   # (B, T, H*W)
    targets_flat = targets.view(B, T, -1)         # (B, T, H*W)
    log_targets_flat = torch.log(targets_flat.clamp_min(1e-8))

    kl_per_step = (targets_flat * (log_targets_flat - log_preds_flat)).sum(dim=-1)  # (B, T)

    return (kl_per_step * valid_mask).sum() / total_valid


def multi_step_ll_score(
    log_preds: torch.Tensor,
    target_fixations: torch.Tensor,
    fixations_len: torch.Tensor,
) -> torch.Tensor:
    """
    Log-likelihood at target fixation locations, averaged over valid steps.

    input:
        log_preds: (B, T, H, W)
        target_fixations: (B, T, 2) in [-1, 1]
        fixations_len: (B,)

    output:
        scalar mean log-likelihood
    """
    B, T, H, W = log_preds.shape
    device = log_preds.device

    valid_mask = (
        torch.arange(T, device=device).unsqueeze(0) < (fixations_len - 1).unsqueeze(1)
    )  # (B, T)

    total_valid = valid_mask.sum()
    if total_valid == 0:
        return torch.tensor(float("nan"), device=device)

    x = target_fixations[..., 0]  # (B, T)
    y = target_fixations[..., 1]  # (B, T)
    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    log_p = log_preds[b_idx, t_idx, row, col]  # (B, T)

    return (log_p * valid_mask).sum() / total_valid


# kept for single-step evaluation (e.g. flattened valid steps passed directly)
def kl_loss(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    KL divergence for a batch of single-step predictions.

    input:
        log_pred: (B, H, W)
        target: (B, H, W)

    output:
        scalar
    """
    B = log_pred.shape[0]
    log_pred_flat = log_pred.view(B, -1)
    target_flat = target.view(B, -1)
    log_target_flat = torch.log(target_flat.clamp_min(1e-8))
    return F.kl_div(log_pred_flat, log_target_flat, reduction="batchmean", log_target=True)


def ll_score(log_pred: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    Log-likelihood at target fixation location for single-step predictions.

    input:
        log_pred: (B, H, W)
        target_xy: (B, 2)

    output:
        scalar
    """
    B, H, W = log_pred.shape
    x = target_xy[:, 0]
    y = target_xy[:, 1]
    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)
    log_p = log_pred[torch.arange(B, device=log_pred.device), row, col]
    return log_p.mean()
