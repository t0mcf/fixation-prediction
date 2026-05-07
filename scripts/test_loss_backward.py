"""test loss computation and backward pass for the full scanpath model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.dataloader import make_dataloader
from src.models.dino_scanpather import ScanpathModel


def kl_loss(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    log_pred: (B, H, W), log-probabilities from the model
    target: (B, H, W), probability target heatmap
    """
    B = log_pred.shape[0]

    log_pred_flat = log_pred.view(B, -1)
    target_flat = target.view(B, -1)

    # target is a probability distribution, so we convert it to log-space.
    log_target_flat = torch.log(target_flat.clamp_min(1e-8))

    return F.kl_div(
        log_pred_flat,
        log_target_flat,
        reduction="batchmean",
        log_target=True,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    loader = make_dataloader(
        split="train",
        batch_size=2,
        num_workers=0,
        max_images=5,
        use_grouped_sampler=False,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )

    batch = next(iter(loader))

    image = batch["image"].to(device)
    prefix = batch["prefix"].to(device)
    prefix_len = batch["prefix_len"].to(device)
    target = batch["heatmap"].to(device)

    model = ScanpathModel(
        visual_encoder="dinov2_base",
        pretrained_encoder=True,
        img_size=224,
        d_model=256,
        n_heads=8,
        scanpath_layers=2,
        max_prefix_len=15,
        heatmap_size=64,
        dropout=0.1,
        decoder_hidden_channels=256,
    ).to(device)

    model.train()

    log_pred = model(image, prefix, prefix_len)

    # log_pred: (B, 64, 64), target: (B, 64, 64)
    print("log_pred shape:", tuple(log_pred.shape), flush=True)
    print("target shape:", tuple(target.shape), flush=True)

    loss = kl_loss(log_pred, target)
    print("loss:", float(loss.detach()), flush=True)

    assert torch.isfinite(loss), "loss is NaN or Inf"

    loss.backward()

    image_encoder_grads = [
        p.grad for p in model.image_encoder.parameters()
        if p.requires_grad
    ]

    trainable_grads = [
        p.grad for p in model.parameters()
        if p.requires_grad
    ]

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_with_grad = sum(g is not None for g in trainable_grads)

    print("trainable params:", num_trainable, flush=True)
    print("trainable tensors with grad:", num_with_grad, "/", len(trainable_grads), flush=True)
    print("image encoder trainable grad tensors:", len(image_encoder_grads), flush=True)

    assert len(image_encoder_grads) == 0, "frozen image encoder should have no trainable gradients"
    assert any(g is not None for g in trainable_grads), "no trainable gradients found"

    print("loss and backward test passed", flush=True)


if __name__ == "__main__":
    main()