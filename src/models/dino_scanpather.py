"""Frozen visual patch encoder based on timm."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


supported_encoders: dict[str, str] = {
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

    optionally accepts visual patch features at each fixation location
    (use_visual_features=True). these are added to the coordinate embedding so
    the transformer reasons over the visual sequence, not just the spatial one:
    each token encodes "i was at (x, y) and saw [semantic content]" rather than
    just "i was at (x, y)".

    input:
        prefix: (B, T, 2)
        prefix_len: (B,)
        patch_feats: (B, T, d_model) or None

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
        use_visual_features: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_prefix_len = max_prefix_len
        self.use_visual_features = use_visual_features

        self.coord_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Embedding(max_prefix_len, d_model)

        if use_visual_features:
            # projects image_adapter output (already d_model) into the token embedding.
            # separate projection rather than direct addition so the network can
            # learn to weight coordinate vs. visual information independently.
            self.patch_feat_proj = nn.Linear(d_model, d_model)

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
        patch_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        prefix: (B, T, 2)
        prefix_len: (B,)
        patch_feats: (B, T, d_model) visual features at each fixation location, or None
        returns: (B, T, d_model)
        """
        _, T, _ = prefix.shape
        device = prefix.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.coord_proj(prefix) + self.pos_embed(positions)

        if self.use_visual_features and patch_feats is not None:
            x = x + self.patch_feat_proj(patch_feats)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device),
            diagonal=1,
        )

        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= prefix_len.unsqueeze(1)
        )

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        # when all tokens in a sample are padding (prefix_len=0), the attention
        # softmax over an all-masked row produces NaN. zero those out — the
        # fusion layer's all_padding guard discards them anyway.
        x = torch.nan_to_num(x, nan=0.0)

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


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    parallel co-attention: image patches and scanpath tokens mutually attend to each other.

    both cross-attention streams are computed in parallel using the pre-update
    representations as keys/values — following the co-attention pattern from LXMERT
    (Tan & Bansal, EMNLP 2019) and ViLBERT (Lu et al., NeurIPS 2019).

    input:
        image_tokens: (B, N, d_model)
        scanpath_tokens: (B, T, d_model)
        prefix_len: (B,)

    output:
        image_tokens_out: (B, N, d_model)
        scanpath_tokens_out: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # image←scanpath: image patches query the scanpath context (existing direction)
        self.img_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        # scanpath←image: fixation tokens query image patch content (new direction)
        self.scan_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )

        self.norm_img_1 = nn.LayerNorm(d_model)
        self.norm_img_2 = nn.LayerNorm(d_model)
        self.norm_scan_1 = nn.LayerNorm(d_model)
        self.norm_scan_2 = nn.LayerNorm(d_model)

        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_scan = nn.Sequential(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        image_tokens: (B, N, d_model)
        scanpath_tokens: (B, T, d_model)
        prefix_len: (B,)
        returns: (image_tokens_out, scanpath_tokens_out)
        """
        _, T, _ = scanpath_tokens.shape
        device = scanpath_tokens.device

        padding_mask = (
            torch.arange(T, device=device).unsqueeze(0) >= prefix_len.unsqueeze(1)
        )
        all_padding = padding_mask.all(dim=1)

        # parallel cross-attention using PRE-update tokens as K, V in both directions
        img_attn_out, _ = self.img_attn(
            query=image_tokens,
            key=scanpath_tokens,
            value=scanpath_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        scan_attn_out, _ = self.scan_attn(
            query=scanpath_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )

        # zero img←scan output for samples where all scanpath positions are padding
        img_attn_out = torch.where(
            all_padding[:, None, None], torch.zeros_like(img_attn_out), img_attn_out,
        )

        image_tokens = self.norm_img_1(image_tokens + img_attn_out)
        scanpath_tokens = self.norm_scan_1(scanpath_tokens + scan_attn_out)

        image_tokens = self.norm_img_2(image_tokens + self.ffn_img(image_tokens))
        scanpath_tokens = self.norm_scan_2(scanpath_tokens + self.ffn_scan(scanpath_tokens))

        return image_tokens, scanpath_tokens


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

            # 16×16 → 32×32: nearest upsample + conv avoids checkerboard artifacts
            # (transposed conv with kernel_size=2 has no kernel overlap → checkerboard)
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout),

            # 32×32 → 64×64
            nn.Upsample(scale_factor=2, mode="nearest"),
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
        decoder_dropout: float | None = None,
        decoder_hidden_channels: int = 256,
        use_visual_scanpath_features: bool = False,
        use_patch_pos_embed: bool = False,
        use_bidirectional_fusion: bool = False,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.use_bidirectional_fusion = use_bidirectional_fusion

        self.image_encoder = FrozenVisualPatchEncoder(
            model_name=visual_encoder,
            pretrained=pretrained_encoder,
            img_size=img_size,
        )

        self.image_adapter = nn.Sequential(
            nn.Linear(self.image_encoder.embed_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # learned 2D patch position embeddings — one vector per grid position,
        # trained end-to-end so the geometry aligns with the scanpath coordinate space.
        # addresses the mismatch between frozen DINOv2 PEs (baked into 768-dim tokens,
        # compressed through the adapter) and the coord_proj space used by the scanpath
        # encoder. indices: row-major, 0 = top-left, num_patches-1 = bottom-right.
        self.use_patch_pos_embed = use_patch_pos_embed
        if use_patch_pos_embed:
            self.patch_pos_embed = nn.Embedding(self.image_encoder.num_patches, d_model)

        self.use_visual_scanpath_features = use_visual_scanpath_features

        self.scanpath_encoder = ScanpathEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=scanpath_layers,
            max_prefix_len=max_seq_len - 1,
            dropout=dropout,
            use_visual_features=use_visual_scanpath_features,
        )

        FusionCls = BidirectionalCrossAttentionFusion if use_bidirectional_fusion else CrossAttentionFusion
        self.fusion = nn.ModuleList([
            FusionCls(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(fusion_layers)
        ])

        # decoder applies Dropout2d which zeroes entire feature channels.
        # at high rates this disrupts the spatial heatmap structure badly,
        # so we expose a separate knob. None = inherit from the global rate.
        effective_decoder_dropout = (
            decoder_dropout if decoder_dropout is not None else dropout
        )
        self.decoder = SpatialHeatmapDecoder(
            d_model=d_model,
            patch_grid_size=self.image_encoder.grid_size,
            heatmap_size=heatmap_size,
            hidden_channels=decoder_hidden_channels,
            dropout=effective_decoder_dropout,
        )

        self.heatmap_size = heatmap_size

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.image_encoder(image)
        adapted = self.image_adapter(patch_tokens)
        if self.use_patch_pos_embed:
            N = adapted.shape[1]
            indices = torch.arange(N, device=adapted.device)
            adapted = adapted + self.patch_pos_embed(indices)
        return adapted

    def forward(
        self,
        image: torch.Tensor,
        fixations: torch.Tensor,
        fixations_len: torch.Tensor,
        ignore_prefix: bool = False,
        paths_per_image: int = 1,
    ) -> torch.Tensor:
        """
        image: (B, 3, 224, 224)
        fixations: (B, max_seq_len, 2)
        fixations_len: (B,)
        paths_per_image: when using the grouped sampler, consecutive groups of
            this size share the same image. pass args.paths_per_image during
            training to run the image encoder only once per unique image instead
            of once per scanpath, giving a paths_per_image-fold speedup on the
            encoder. leave at 1 (default) during validation/inference.
        returns: (B, max_seq_len - 1, heatmap_size, heatmap_size)
        """
        B = image.shape[0]
        T = fixations.shape[1] - 1  # number of prediction steps

        if paths_per_image > 1 and B % paths_per_image == 0:
            # grouped sampler: image[0:P] are identical, image[P:2P] identical, etc.
            # run the encoder only on the unique images then tile the result.
            unique_images = image[::paths_per_image]           # (B//P, 3, H, W)
            image_tokens = self.encode_image(unique_images).repeat_interleave(
                paths_per_image, dim=0
            )                                                  # (B, N, d_model)
        else:
            image_tokens = self.encode_image(image)            # (B, N, d_model)

        if ignore_prefix:
            logit = self.decoder(image_tokens)  # (B, H, W)
            log_h = F.log_softmax(logit.view(B, -1), dim=-1).view_as(logit)
            return log_h.unsqueeze(1).expand(-1, T, -1, -1)

        prefix = fixations[:, :-1, :]  # (B, T, 2)
        prefix_len = (fixations_len - 1).clamp(min=0)  # (B,)

        # look up image patch features at each fixation location so the scanpath
        # encoder knows not just where each fixation was, but what was seen there.
        patch_feats = None
        if self.use_visual_scanpath_features:
            gs = self.image_encoder.grid_size  # e.g. 16 for dinov2_base
            # map [-1, 1] coordinates to patch grid indices [0, gs-1]
            col = ((prefix[..., 0] + 1.0) / 2.0 * (gs - 1)).round().long().clamp(0, gs - 1)
            row = ((prefix[..., 1] + 1.0) / 2.0 * (gs - 1)).round().long().clamp(0, gs - 1)
            patch_idx = row * gs + col  # (B, T) row-major patch index
            b_idx = torch.arange(B, device=image.device).unsqueeze(1).expand(B, T)
            patch_feats = image_tokens[b_idx, patch_idx]  # (B, T, d_model)

        scanpath_tokens = self.scanpath_encoder(prefix, prefix_len, patch_feats)  # (B, T, d_model)

        # eff_len[b, t] = min(t+1, prefix_len[b]):
        # encodes the causal constraint (only tokens 0..t visible at step t)
        # and variable sequence length in one tensor.
        t_idx = torch.arange(T, device=image.device)
        eff_len_flat = torch.min(
            prefix_len.unsqueeze(1),       # (B, 1)
            (t_idx + 1).unsqueeze(0),      # (1, T)
        ).reshape(B * T)                   # (B*T,)

        # tile inputs across T so all steps run in one batched forward pass
        # instead of a Python loop. Both fusion classes are agnostic to batch size,
        # so passing B*T as the batch dimension works without any changes there.
        image_tiled = image_tokens.repeat_interleave(T, dim=0)    # (B*T, N, d_model)
        scan_tiled = scanpath_tokens.repeat_interleave(T, dim=0)  # (B*T, T, d_model)

        fused = image_tiled
        scan_fused = scan_tiled
        for layer in self.fusion:
            if self.use_bidirectional_fusion:
                fused, scan_fused = layer(fused, scan_fused, eff_len_flat)
            else:
                fused = layer(fused, scan_fused, eff_len_flat)

        logit = self.decoder(fused)  # (B*T, H, W)
        log_h = F.log_softmax(logit.view(B * T, -1), dim=-1).view_as(logit)

        return log_h.view(B, T, self.heatmap_size, self.heatmap_size)
