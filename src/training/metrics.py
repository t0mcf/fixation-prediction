"""evaluation metrics for scanpath prediction."""

from __future__ import annotations

import torch


def nss_score(log_pred: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    normalized scanpath saliency at the target fixation location.

    input:
        log_pred: (B, H, W)
        target_xy: (B, 2)

    output:
        nss: scalar tensor
    """
    pred = log_pred.exp()
    B, H, W = pred.shape
    device = pred.device

    x = target_xy[:, 0]
    y = target_xy[:, 1]

    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    pred_flat = pred.view(B, -1)
    mean = pred_flat.mean(dim=1).view(B, 1, 1)
    std = pred_flat.std(dim=1).view(B, 1, 1).clamp_min(1e-8)

    pred_norm = (pred - mean) / std
    values = pred_norm[torch.arange(B, device=device), row, col]

    return values.mean()