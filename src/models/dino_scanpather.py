"""Frozen visual patch encoder based on timm."""

from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn


supported_encoders: Dict[str, str] = {
    "dinov2_base": "vit_base_patch14_dinov2",
    "dinov2_base_reg": "vit_base_patch14_reg4_dinov2",
    "dinov2_large": "vit_large_patch14_dinov2",
}


class FrozenVisualPatchEncoder(nn.Module):
    """
    frozen visual encoder that returns spatial patch tokens.

    architecture:
        image -> timm vision transformer -> discard cls/register tokens
        -> keep spatial patch tokens only
        
    input:
        image: (B, 3, H, W)

    output:
        patch_tokens: (B, N, D)

    for dinov2 vit-b/14 with 224x224 input:
        N = 16 * 16 = 256 patches
        D = 768 embedding dimensions
    """

    def __init__(
        self,
        model_name: str = "dinov2_base",
        pretrained: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()

        resolved_name = supported_encoders.get(model_name, model_name)

        self.vit = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            img_size=img_size,
        )

        self.model_name = resolved_name
        self.embed_dim = self.vit.embed_dim

        patch_size = self.vit.patch_embed.patch_size
        patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size

        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, H, W)
        returns: (B, N, D), where N is the number of spatial patch tokens
        """
        self.vit.eval()

        features = self.vit.forward_features(image)

        # timm ViTs may return cls/register tokens before the spatial patch tokens.
        # the last N tokens are the spatial patch tokens.
        patch_tokens = features[:, -self.num_patches :, :]

        return patch_tokens
    
    
class ScanpathEncoder(nn.Module):
    """
    encodes previous fixations into scanpath tokens.
    
    architecture:
        raw (x, y) coordinates -> linear projection + learned positional embeddings
        -> causal transformer encoder -> scanpath tokens

    input:
        prefix: (B, T, 2)
        prefix_len: (B,)

    output:
        scanpath_tokens: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        max_prefix_len: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_prefix_len = max_prefix_len

        self.coord_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Embedding(max_prefix_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        prefix: torch.Tensor,
        prefix_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        prefix: (B, T, 2)
        prefix_len: (B,)
        returns: (B, T, d_model)
        """
        _, T, _ = prefix.shape
        device = prefix.device

        positions = torch.arange(T, device=device).unsqueeze(0)

        # x: (B, T, d_model)
        x = self.coord_proj(prefix) + self.pos_embed(positions)

        # causal_mask[i, j] = true means token i cannot attend to token j.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device),
            diagonal=1,
        )

        # padding_mask[b, j] = true means token j is padding for sample b.
        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= prefix_len.unsqueeze(1)
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        return self.norm(x)
    
    
class CrossAttentionFusion(nn.Module):
    """
    fuses image patch tokens with scanpath history tokens.

    architecture:
        image patch tokens are used as queries.
        scanpath tokens are used as keys and values.
        cross-attention updates each image patch with history information.
        a small feed-forward block is applied afterwards.

    input:
        image_tokens: (B, N, d_model)
        scanpath_tokens: (B, T, d_model)
        prefix_len: (B,)

    output:
        fused_tokens: (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        scanpath_tokens: torch.Tensor,
        prefix_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_tokens: (B, N, d_model)
        scanpath_tokens: (B, T, d_model)
        prefix_len: (B,)
        returns: (B, N, d_model)
        """
        _, T, _ = scanpath_tokens.shape
        device = scanpath_tokens.device

        # padding_mask[b, j] = true means scanpath token j is padding for sample b.
        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= prefix_len.unsqueeze(1)
        )

        # attn_out: (B, N, d_model)
        attn_out, _ = self.cross_attn(
            query=image_tokens,
            key=scanpath_tokens,
            value=scanpath_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        # residual connection keeps the original visual patch information.
        x = self.norm1(image_tokens + attn_out)

        # feed-forward block updates each patch token independently.
        x = self.norm2(x + self.ffn(x))

        return x