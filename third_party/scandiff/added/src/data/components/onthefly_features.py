"""Wrapper that replaces precomputed .pth features with on-the-fly DINOv2 extraction.

Usage: wrap any existing dataset (MIT1003Dataset, COCOFreeViewDataset, etc.)
    dataset = OnTheFlyFeatureDataset(base_dataset, images_dir="./data/oursynth_10k/images")

The wrapper is transparent: it returns the same dict as the base dataset,
but replaces the 'img' field with freshly extracted DINOv2 features.
The base dataset class and its code are NOT modified.
"""

import torch
import timm
from pathlib import Path
from PIL import Image


class OnTheFlyFeatureDataset:
    def __init__(self, base_dataset, images_dir, device="cpu"):
        self.base = base_dataset
        self.images_dir = Path(images_dir)
        self.device = device

        self.model = timm.create_model(
            'vit_base_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
        ).eval().to(device)

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        for p in self.model.parameters():
            p.requires_grad = False

    def _resolve_path(self, img_filename):
        """Try direct path first, then search for case-insensitive match."""
        path = self.images_dir / img_filename
        if path.exists():
            return path
        # oursynth uses "classname_id.jpeg" but ImageNet has "classname/classname_id.JPEG"
        stem = Path(img_filename).stem
        # extract class dir from filename (e.g. n01440764_10026 -> n01440764)
        class_dir = stem.rsplit("_", 1)[0]
        for ext in [".JPEG", ".jpeg", ".jpg", ".png"]:
            candidate = self.images_dir / class_dir / (stem + ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find image for {img_filename} in {self.images_dir}")

    @torch.no_grad()
    def _extract(self, img_filename):
        path = self._resolve_path(img_filename)
        img = Image.open(path).convert("RGB").resize((518, 518))
        x = self.transform(img).unsqueeze(0).to(self.device)
        feats = self.model.forward_features(x)
        feats = feats[:, 5:, :]  # remove CLS + 4 register tokens
        return feats.squeeze(0).cpu().float()

    def __getitem__(self, index):
        item = self.base[index]
        item['img'] = self._extract(item['img_filename'])
        return item

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)
