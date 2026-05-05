"""test the frozen visual patch encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.models.dino_scanpather import FrozenVisualPatchEncoder


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    encoder = FrozenVisualPatchEncoder(
        model_name="dinov2_base",
        pretrained=True,
        img_size=224,
    ).to(device)

    print("model name:", encoder.model_name, flush=True)
    print("patch size:", encoder.patch_size, flush=True)
    print("grid size:", encoder.grid_size, flush=True)
    print("num patches:", encoder.num_patches, flush=True)
    print("embed dim:", encoder.embed_dim, flush=True)

    image = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        patch_tokens = encoder(image)

    # patch_tokens: (B, N, D)
    print("patch tokens shape:", tuple(patch_tokens.shape), flush=True)

    expected_shape = (2, encoder.num_patches, encoder.embed_dim)
    assert tuple(patch_tokens.shape) == expected_shape, (
        f"expected {expected_shape}, got {tuple(patch_tokens.shape)}"
    )

    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())

    print("total params:", total_params, flush=True)
    print("trainable params:", trainable_params, flush=True)

    assert trainable_params == 0, "visual encoder should be frozen"

    print("visual encoder test passed", flush=True)


if __name__ == "__main__":
    main()