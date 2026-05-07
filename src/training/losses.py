"""training losses for scanpath prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_loss(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    kl divergence to gaussian target heatmap.

    input:
        log_pred: (B, H, W)
        target: (B, H, W)

    output:
        loss: scalar tensor
    """
    B = log_pred.shape[0]

    log_pred_flat = log_pred.view(B, -1)
    target_flat = target.view(B, -1)
    log_target_flat = torch.log(target_flat.clamp_min(1e-8))

    return F.kl_div(
        log_pred_flat,
        log_target_flat,
        reduction="batchmean",
        log_target=True,
    )


def nll_loss(log_pred: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    negative log likelihood at the target fixation location.

    input:
        log_pred: (B, H, W)
        target_xy: (B, 2)

    output:
        loss: scalar tensor
    """
    B, H, W = log_pred.shape
    device = log_pred.device

    x = target_xy[:, 0]
    y = target_xy[:, 1]

    col = torch.round((x + 1.0) / 2.0 * (W - 1)).long().clamp(0, W - 1)
    row = torch.round((y + 1.0) / 2.0 * (H - 1)).long().clamp(0, H - 1)

    log_p = log_pred[torch.arange(B, device=device), row, col]

    return -log_p.mean()