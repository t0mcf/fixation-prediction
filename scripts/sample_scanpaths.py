"""
Autoregressive scanpath sampling from our next-fixation model.

Our ScanpathModel emits, for a prefix f_0..f_t, a 64x64 log-prob heatmap for the
next fixation f_{t+1}. To produce a FULL scanpath we sample step by step:

  start with f_0 = (0,0) center  (matches the training convention: every scanpath
  begins at the central fixation), then for m = 1..L-1:
    - put context f_0..f_{m-1} at positions 0..m-1, set fixations_len = m+1
    - forward -> read heatmap at step (m-1)  [predicts f_m given f_0..f_{m-1}]
    - temperature-sample a 64x64 cell, convert to (x,y) in [-1,1] (+half-cell jitter)
    - append as f_m

This puts our (probabilistic, next-fixation) model on a generative footing so its
scanpaths can be scored with sequence metrics (MultiMatch / ScanMatch) — the same
metrics ScanDiff reports. The model has no termination head, so scanpath LENGTHS are
provided (e.g. matched to the human scanpaths we compare against).

Coordinates are in [-1,1] (model space). Convert to pixels at scoring time with the
image's (W,H): px = (x+1)/2*(W-1).

Usage (quick test):
  python scripts/sample_scanpaths.py \
    --checkpoint runs/scale_800k_5p_ll/checkpoints/best_val_nss.pt \
    --images <img1.jpeg> <img2.jpeg> --per-image 3 --length 10 --out /tmp/sp.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.dino_scanpather import ScanpathModel
from src.data.constants import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def build_model(ckpt: dict, device: str) -> ScanpathModel:
    a = ckpt.get("args", {})
    m = ScanpathModel(
        visual_encoder=a.get("visual_encoder", "dinov2_base"),
        pretrained_encoder=not a.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=a.get("d_model", 256), n_heads=a.get("n_heads", 8),
        scanpath_layers=a.get("scanpath_layers", 2),
        fusion_layers=a.get("fusion_layers", 1),
        max_seq_len=a.get("max_seq_len", 16),
        heatmap_size=a.get("heatmap_size", 64),
        dropout=0.0, decoder_dropout=0.0,
        decoder_hidden_channels=a.get("decoder_hidden_channels", 256),
        decoder_upsample=a.get("decoder_upsample", "nearest"),

        use_visual_scanpath_features=a.get("visual_scanpath_features", False),
        use_patch_pos_embed=a.get("patch_pos_embed", False),
        use_bidirectional_fusion=a.get("bidirectional_fusion", False),
        use_fixation_query_fusion=a.get("fixation_query_fusion", False),
    ).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def sample_scanpaths(
    model: ScanpathModel,
    images: torch.Tensor,        # (B, 3, 224, 224) normalised
    lengths: Sequence[int],      # (B,) total fixations per scanpath, incl. center start
    device: str,
    temperature: float = 1.0,
    jitter: float = 1.0,         # half-cell uniform jitter scale (0 = cell centers)
    generator: torch.Generator | None = None,
) -> list[np.ndarray]:
    """Returns a list of (L_b, 2) arrays in [-1,1]; row 0 is the (0,0) center."""
    B = images.shape[0]
    S = model.max_seq_len
    H = W = model.heatmap_size
    lengths = [min(int(l), S) for l in lengths]
    maxL = max(lengths)

    fix = torch.zeros(B, S, 2, device=device)        # f_0 = (0,0) center (zeros)
    half = 1.0 / (W - 1)                              # half-cell in [-1,1] coords

    # encode each image ONCE (frozen DINOv2) and reuse across all steps
    image_tokens = model.encode_image(images)        # (B, N, d_model)

    for m in range(1, maxL):
        flen = torch.full((B,), m + 1, dtype=torch.long, device=device)
        logh = model(images, fix, flen, image_tokens=image_tokens)  # (B, S-1, H, W)
        logp = logh[:, m - 1].reshape(B, H * W)      # next-fixation log-prob
        probs = torch.softmax(logp / temperature, dim=-1)
        idx = torch.multinomial(probs, 1, generator=generator).squeeze(1)  # (B,)
        row = (idx // W).float()
        col = (idx % W).float()
        x = col / (W - 1) * 2.0 - 1.0
        y = row / (H - 1) * 2.0 - 1.0
        if jitter > 0:
            jx = (torch.rand(B, device=device, generator=generator) * 2 - 1) * half * jitter
            jy = (torch.rand(B, device=device, generator=generator) * 2 - 1) * half * jitter
            x = (x + jx).clamp(-1, 1)
            y = (y + jy).clamp(-1, 1)
        fix[:, m, 0] = x
        fix[:, m, 1] = y

    out = []
    for b in range(B):
        out.append(fix[b, : lengths[b]].cpu().numpy())
    return out


def load_image(path: str) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T
    import os
    # Geometry must match how the checkpoint was TRAINED (same convention as
    # src/data/dataset.py): OURS_CROP_TRANSFORM=1 -> Resize(256)+CenterCrop(224)
    # (protocol-v2 ladder), unset -> direct stretch (MIT1003 transfer usage).
    if os.environ.get("OURS_CROP_TRANSFORM") == "1":
        resize = [T.Resize(256), T.CenterCrop(224)]
    else:
        resize = [T.Resize((224, 224))]
    tf = T.Compose(resize + [T.ToTensor(),
                   T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return tf(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image file paths")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--length", type=int, default=10, help="total fixations incl. center")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(ckpt, args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    # build a batch: each image repeated per-image times
    imgs, owners = [], []
    for i, p in enumerate(args.images):
        t = load_image(p)
        for _ in range(args.per_image):
            imgs.append(t); owners.append(p)
    batch = torch.stack(imgs).to(args.device)
    lengths = [args.length] * len(imgs)

    sps = sample_scanpaths(model, batch, lengths, args.device,
                           temperature=args.temperature, jitter=args.jitter, generator=gen)

    print(f"generated {len(sps)} scanpaths from {len(args.images)} images")
    s0 = sps[0]
    print(f"example scanpath shape {s0.shape}; first 4 (x,y) in [-1,1]:")
    print(np.round(s0[:4], 3))
    print(f"  start at center? f0={np.round(s0[0],3)} (expect [0,0])")
    print(f"  coord ranges over all: x[{min(p[:,0].min() for p in sps):.2f},"
          f"{max(p[:,0].max() for p in sps):.2f}] "
          f"y[{min(p[:,1].min() for p in sps):.2f},{max(p[:,1].max() for p in sps):.2f}]")
    # mean step size (sanity: should be > 0, scanpaths move)
    steps = np.mean([np.linalg.norm(np.diff(p, axis=0), axis=1).mean() for p in sps])
    print(f"  mean saccade length (norm coords): {steps:.3f}")

    if args.out:
        np.savez(args.out, scanpaths=np.array(sps, dtype=object), owners=np.array(owners))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
