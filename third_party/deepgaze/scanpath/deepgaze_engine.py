"""
scanpath/deepgaze_engine.py
===========================
Model builders, data-loading utilities, and MIT1003 conversion helpers.
All training modes (imagenet, salicon, mit1003) import from here.
"""
import os
import sys
import shutil
from collections import OrderedDict
from pathlib import Path

# Make models/DeepGaze importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "DeepGaze"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from imageio.v3 import imread, imwrite
from tqdm import tqdm
from torchvision import transforms
import pysaliency

# NumPy 2.0 compatibility
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from deepgaze_pytorch.layers import (
    Bias, Conv2dMultiInput, FlexibleScanpathHistoryEncoding,
    LayerNorm, LayerNormMultiInput,
)
from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
from deepgaze_pytorch.features.densenet import RGBDenseNet201
from deepgaze_pytorch.data import (
    ImageDataset, ImageDatasetSampler, FixationDataset, FixationMaskTransform,
)

# ---------------------------------------------------------------------------
# Image resize pipeline (used for ImageNet / synthetic scanpaths)
# ---------------------------------------------------------------------------

IMAGENET_SIZE = 224

standard_resize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGENET_SIZE),
])

_fixation_mask_transform = FixationMaskTransform(sparse=False)


class ResizedFileStimuli(pysaliency.FileStimuli):
    """FileStimuli subclass that lazily resizes every image to IMAGENET_SIZE×IMAGENET_SIZE.

    This gives the same per-image result as ``preprocess_to_inmemory`` but
    *without* holding everything in RAM.  Each DataLoader worker reads the file,
    resizes it, and returns the uint8 numpy array — exactly what
    ``ImageDataset._get_image_data`` expects.
    """

    def __init__(self, filenames, resize_transform=None, **kwargs):
        self._resize = resize_transform or standard_resize
        target_shape = (IMAGENET_SIZE, IMAGENET_SIZE, 3)
        shapes = [target_shape] * len(filenames)
        # cached=False → do not keep all images in memory (that's the point)
        super().__init__(filenames=list(filenames), cached=False, shapes=shapes, **kwargs)

    def load_stimulus(self, n):
        img = Image.open(self.filenames[n]).convert('RGB')
        img = self._resize(img)
        return np.array(img, dtype=np.uint8)


def preprocess_to_inmemory(stimuli) -> pysaliency.Stimuli:
    """Load all FileStimuli once, resize to 224×224, return an in-memory pysaliency.Stimuli.
    All DataLoader workers then access pre-sized uint8 arrays via fork — zero per-sample disk I/O.
    Pre-allocates the output array upfront to avoid the 2× RAM peak from list-then-stack.
    """
    n = len(stimuli.filenames)
    out = np.empty((n, IMAGENET_SIZE, IMAGENET_SIZE, 3), dtype=np.uint8)
    for i, filename in enumerate(tqdm(stimuli.filenames, desc="Pre-loading images", leave=False)):
        img = Image.open(filename).convert('RGB')
        out[i] = np.array(standard_resize(img), dtype=np.uint8)
    return pysaliency.Stimuli(out)


def preprocess_to_resized_filestimuli(stimuli) -> ResizedFileStimuli:
    """Wrap FileStimuli so every load lazily resizes to IMAGENET_SIZE×IMAGENET_SIZE.
    No RAM cost — images are read and resized on the fly per batch.
    """
    return ResizedFileStimuli(stimuli.filenames)


def sample_transform(data):
    """Images arrive as float32 CHW — only fixation mask creation needed."""
    return _fixation_mask_transform(data)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_saliency_network(input_channels: int = 2048):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0',      nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0',      Bias(8)),
        ('softplus0',  nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1',      nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1',      Bias(16)),
        ('softplus1',  nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2',      nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2',      Bias(1)),
        ('softplus2',  nn.Softplus()),
    ]))


def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(
            in_fixations=4, channels_per_fixation=3,
            out_channels=128, kernel_size=[1, 1], bias=True,
        )),
        ('softplus0',  nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1',      nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1',      Bias(16)),
        ('softplus1',  nn.Softplus()),
    ]))


def build_fixation_selection_network(scanpath_features: int = 16):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, scanpath_features])),
        ('conv0',      Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
        ('bias0',      Bias(128)),
        ('softplus0',  nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1',      nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1',      Bias(16)),
        ('softplus1',  nn.Softplus()),

        ('conv2',      nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))


def _make_backbone(cfg):
    """Backbone-swap ablation switch: model.backbone = densenet (default) | dinov2.
    Returns (features_module, saliency_input_channels)."""
    backbone = getattr(cfg.model, "backbone", None) or "densenet"
    if backbone == "densenet":
        return FeatureExtractor(RGBDenseNet201(), list(cfg.model.densenet_layers)), 2048
    if backbone == "dinov2":
        from deepgaze_pytorch.features.dinov2 import RGBDINOv2
        return RGBDINOv2(), 768
    raise ValueError(f"unknown model.backbone: {backbone}")


def make_spatial_model(cfg, device: str) -> DeepGazeIII:
    """Spatial-only DeepGazeIII (no scanpath history). Uses stage1 config for architecture."""
    s = cfg.training.stage1
    features, sal_ch = _make_backbone(cfg)
    return DeepGazeIII(
        features=features,
        saliency_network=build_saliency_network(sal_ch),
        scanpath_network=None,
        fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
        downsample=s.downsample,
        readout_factor=s.readout_factor,
        saliency_map_factor=s.saliency_map_factor,
        included_fixations=[],
    ).to(device)


def make_scanpath_model(cfg, stage_key: str, device: str) -> DeepGazeIII:
    """Full DeepGazeIII with scanpath history. Uses the given stage config for architecture."""
    s = getattr(cfg.training, stage_key)
    features, sal_ch = _make_backbone(cfg)
    return DeepGazeIII(
        features=features,
        saliency_network=build_saliency_network(sal_ch),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=s.downsample,
        readout_factor=s.readout_factor,
        saliency_map_factor=s.saliency_map_factor,
        included_fixations=[-1, -2, -3, -4],
    ).to(device)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def _make_loader(dataset_cls, stimuli, fixations, centerbias, batch_size,
                 num_workers, lmdb_path=None, **dataset_kwargs):
    """Internal factory. Extra kwargs are forwarded to the dataset constructor."""
    if lmdb_path is not None:
        Path(lmdb_path).mkdir(parents=True, exist_ok=True)

    dataset = dataset_cls(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=sample_transform,
        average='image',
        lmdb_path=lmdb_path,
        **dataset_kwargs,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def make_spatial_loader(stimuli, fixations, centerbias, batch_size, num_workers,
                        lmdb_path=None, **kwargs):
    """ImageDataset loader.  Pass cached=True to force in-memory caching (imagenet mode)."""
    return _make_loader(
        ImageDataset, stimuli, fixations, centerbias, batch_size, num_workers,
        lmdb_path=lmdb_path,
        **kwargs,
    )


def make_scanpath_loader(stimuli, fixations, centerbias, batch_size, num_workers,
                         lmdb_path=None, **kwargs):
    """FixationDataset loader with 4-step history.
    Pass cache_image_data=True for small in-memory datasets (imagenet mode).
    """
    return _make_loader(
        FixationDataset, stimuli, fixations, centerbias, batch_size, num_workers,
        lmdb_path=lmdb_path,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        **kwargs,
    )


def make_scheduler(optimizer, milestones: list):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)


# ---------------------------------------------------------------------------
# MIT1003 two-size conversion utilities
# ---------------------------------------------------------------------------

def _twosize_target(h: int, w: int):
    """Canonical (H, W) for MIT1003 two-size normalisation."""
    return (768, 1024) if h < w else (1024, 768)


def convert_fixation_trains(stimuli, fixations) -> pysaliency.FixationTrains:
    """Scale all fixation coordinates to match the two-size target resolution."""
    train_xs = fixations.train_xs.copy()
    train_ys = fixations.train_ys.copy()

    for i in tqdm(range(len(train_xs)), desc="Converting fixation trains", leave=False):
        n = fixations.train_ns[i]
        h, w = stimuli.shapes[n][:2]
        new_h, new_w = _twosize_target(h, w)
        train_xs[i] *= new_w / w
        train_ys[i] *= new_h / h

    return pysaliency.FixationTrains(
        train_xs=train_xs,
        train_ys=train_ys,
        train_ts=fixations.train_ts.copy(),
        train_ns=fixations.train_ns.copy(),
        train_subjects=fixations.train_subjects.copy(),
        attributes={
            key: getattr(fixations, key).copy()
            for key in fixations.__attributes__
            if key not in ['subjects', 'scanpath_index']
        },
    )


def convert_stimuli(stimuli: pysaliency.FileStimuli, dest_dir: Path) -> pysaliency.FileStimuli:
    """Resize all stimuli to their canonical two-size resolution; skip if already correct."""
    assert isinstance(stimuli, pysaliency.FileStimuli)
    stimuli_dir = dest_dir / 'stimuli'
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    new_filenames = []
    for filename in tqdm(stimuli.filenames, desc="Resizing stimuli", leave=False):
        src = imread(filename)
        h, w = src.shape[:2]
        new_h, new_w = _twosize_target(h, w)
        basename = os.path.basename(str(filename))
        new_path = stimuli_dir / basename

        # idempotent: reuse an existing converted file so concurrent CV fold
        # jobs sharing one out_dir never write the same image twice
        if new_path.exists():
            new_filenames.append(new_path)
            continue

        if (h, w) != (new_h, new_w):
            dst = np.array(Image.fromarray(src).resize((new_w, new_h), Image.BILINEAR))
            imwrite(new_path, dst)
        else:
            shutil.copy(str(filename), new_path)

        new_filenames.append(new_path)

    return pysaliency.FileStimuli(new_filenames)
