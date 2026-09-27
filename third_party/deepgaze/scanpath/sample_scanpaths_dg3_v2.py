"""DG3 whole-scanpath sampling on the protocol-v2 scoring subsets (ImageNet).

Thin wrapper around scanpath.sample_scanpaths_dg3 (untouched: it is a frozen
dependency of the 2026-09 test manifests) that ONLY changes the image
geometry to the v2 protocol: Resize(shorter side 256) + CenterCrop(224),
i.e. the frame the AV generator, our model, DG3's synthetic training
(deepgaze_engine.standard_resize) and ScanDiff's features all use. The
MIT zero-shot sampler resizes directly to 224x224, which would misplace
every coordinate on ImageNet stimuli.

Everything else is identical: flat centerbias, last-4 history with NaN for
missing early steps, multinomial sampling from the 224x224 density at
temperature T, one sampled path per reference path with the reference's
length (16), centre start; output {stem: [(L,2) in [-1,1]]} for
scandiff/tools/score_scanpaths.py.

  python -m scanpath.sample_scanpaths_dg3_v2 --checkpoint ... \
      --spec <subset>/subset_gt_5ref.json --stimuli-dir <subset>/stimuli \
      --ext .JPEG --out <subset>/gen/dg3_v2_<scale>_best_T1/gen.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from scanpath import sample_scanpaths_dg3 as base

_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(base.SIZE)])


def load_image_crop(path: Path) -> torch.Tensor:
    """-> (3, 224, 224) float32 in [0,255] under the v2 crop geometry."""
    im = _crop(Image.open(path).convert("RGB"))
    arr = np.asarray(im, dtype=np.float32)
    assert arr.shape == (base.SIZE, base.SIZE, 3), arr.shape
    return torch.from_numpy(arr.transpose(2, 0, 1))


if __name__ == "__main__":
    base.load_image = load_image_crop     # the only behavioural change
    print("sample_scanpaths_dg3_v2: image geometry = Resize(256)+CenterCrop(224)", flush=True)
    base.main()
