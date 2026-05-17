"""Frozen visual patch encoder based on timm."""

from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # timm ViTs return cls/register tokens before spatial patch tokens.
        # the last N tokens are always the spatial patch tokens.
        patch_tokens = features[:, -self.num_patches:, :]

        return patch_tokens


class ScanpathEncoder(nn.Module):
    """
    encodes previous fixations into scanpath tokens.

    architecture:
        raw (x, y) coordinates -> linear projection + learned positional embeddings
        -> causal transformer encoder -> scanpath tokens

    the causal mask is necessary for correctness in predict-all training:
        token[t] may only attend to fixations 0..t, so that the prediction
        at step t is not contaminated by future fixations.

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
        x = self.coord_proj(prefix) + self.pos_embed(positions)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device),
            diagonal=1,
        )

        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= prefix_len.unsqueeze(1)
        )

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

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

        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= prefix_len.unsqueeze(1)
        )

        all_padding = padding_mask.all(dim=1)

        attn_out, _ = self.cross_attn(
            query=image_tokens,
            key=scanpath_tokens,
            value=scanpath_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        attn_out = torch.where(
            all_padding[:, None, None],
            torch.zeros_like(attn_out),
            attn_out,
        )

        x = self.norm1(image_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x


class SpatialHeatmapDecoder(nn.Module):
    """
    decodes history-conditioned patch tokens into spatial heatmap logits.

    architecture:
        patch tokens are reshaped back into a 2d patch grid.
        convolutional layers and bilinear upsampling convert the patch grid
        into heatmap logits at the target resolution.

    input:
        patch_tokens: (B, N, d_model)

    output:
        logits: (B, heatmap_size, heatmap_size)
    """

    def __init__(
        self,
        d_model: int = 256,
        patch_grid_size: int = 16,
        heatmap_size: int = 64,
        hidden_channels: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_grid_size = patch_grid_size
        self.heatmap_size = heatmap_size

        self.net = nn.Sequential(
            nn.Conv2d(d_model, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels // 4),
            nn.GELU(),

            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, N, d_model)
        returns: (B, heatmap_size, heatmap_size)
        """
        B, N, d_model = patch_tokens.shape
        expected_n = self.patch_grid_size * self.patch_grid_size

        if N != expected_n:
            raise ValueError(f"expected {expected_n} patch tokens, got {N}")

        x = patch_tokens.transpose(1, 2).reshape(
            B, d_model, self.patch_grid_size, self.patch_grid_size,
        )

        return self.net(x).squeeze(1)


class ScanpathModel(nn.Module):
    """
    dino-based model for predict-all next-fixation prediction.

    for each fixation step t in [0, N-2], predicts the heatmap for fixation t+1
    given image and fixations[0..t] as context. all predictions are produced in
    a single forward pass via a loop over t with per-step cross-attention masking.

    input:
        image: (B, 3, 224, 224)
        fixations: (B, max_seq_len, 2)   full padded fixation sequence
        fixations_len: (B,)              actual sequence length per sample

    output:
        log_heatmaps: (B, max_seq_len - 1, heatmap_size, heatmap_size)
                      log-probability heatmap for each prediction step.
                      step t is valid for sample b when t < fixations_len[b] - 1.
    """

    def __init__(
        self,
        visual_encoder: str = "dinov2_base",
        pretrained_encoder: bool = True,
        img_size: int = 224,
        d_model: int = 256,
        n_heads: int = 8,
        scanpath_layers: int = 2,
        fusion_layers: int = 1,
        max_seq_len: int = 16,
        heatmap_size: int = 64,
        dropout: float = 0.1,
        decoder_hidden_channels: int = 256,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len

        self.image_encoder = FrozenVisualPatchEncoder(
            model_name=visual_encoder,
            pretrained=pretrained_encoder,
            img_size=img_size,
        )

        self.image_adapter = nn.Sequential(
            nn.Linear(self.image_encoder.embed_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.scanpath_encoder = ScanpathEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=scanpath_layers,
            max_prefix_len=max_seq_len - 1,
            dropout=dropout,
        )

        self.fusion = nn.ModuleList([
            CrossAttentionFusion(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(fusion_layers)
        ])

        self.decoder = SpatialHeatmapDecoder(
            d_model=d_model,
            patch_grid_size=self.image_encoder.grid_size,
            heatmap_size=heatmap_size,
            hidden_channels=decoder_hidden_channels,
            dropout=dropout,
        )

        self.heatmap_size = heatmap_size

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.image_encoder(image)
        return self.image_adapter(patch_tokens)

    def forward(
        self,
        image: torch.Tensor,
        fixations: torch.Tensor,
        fixations_len: torch.Tensor,
        ignore_prefix: bool = False,
    ) -> torch.Tensor:
        """
        image: (B, 3, 224, 224)
        fixations: (B, max_seq_len, 2)
        fixations_len: (B,)
        returns: (B, max_seq_len - 1, heatmap_size, heatmap_size)
        """
        B = image.shape[0]
        T = fixations.shape[1] - 1  # number of prediction steps

        image_tokens = self.encode_image(image)  # (B, N, d_model)

        if ignore_prefix:
            logit = self.decoder(image_tokens)  # (B, H, W)
            log_h = F.log_softmax(logit.view(B, -1), dim=-1).view_as(logit)
            return log_h.unsqueeze(1).expand(-1, T, -1, -1)

        # prefix: all fixations except the last  (B, T, 2)
        prefix = fixations[:, :-1, :]
        prefix_len = (fixations_len - 1).clamp(min=0)  # (B,)

        # encode full prefix once with causal mask;
        # token[t] encodes "given fixations 0..t" and is used to predict fixation t+1
        scanpath_tokens = self.scanpath_encoder(prefix, prefix_len)  # (B, T, d_model)

        heatmaps = []
        for t in range(T):
            # for step t, each sample may use at most t+1 scanpath tokens,
            # and also no more than its actual valid prefix length
            eff_len = torch.clamp(prefix_len, max=t + 1)  # (B,)

            fused = image_tokens
            for layer in self.fusion:
                fused = layer(fused, scanpath_tokens, eff_len)

            logit = self.decoder(fused)  # (B, H, W)
            log_h = F.log_softmax(logit.view(B, -1), dim=-1).view_as(logit)
            heatmaps.append(log_h)

        return torch.stack(heatmaps, dim=1)  # (B, T, H, W)
