"""Shared constants for data loading and visualisation."""

# ImageNet normalisation statistics — used by all DINOv2 variants in timm.
# Verified against timm.get_pretrained_cfg() for:
#   vit_base_patch14_dinov2, vit_base_patch14_reg4_dinov2, vit_large_patch14_dinov2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
