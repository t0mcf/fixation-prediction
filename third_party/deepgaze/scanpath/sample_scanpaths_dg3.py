"""
Autoregressive scanpath sampling from a DeepGaze III model, in the same format
and under the same protocol as fixation-prediction/scripts/sample_for_eval.py,
so DG3 can be scored on MultiMatch / ScanMatch next to our model and ScanDiff.

Protocol (matched to our model's realism ladder):
  * zero-shot: the synthetic-trained DG3 ladder checkpoint is applied to MIT1003
    with NO fine-tuning (runs/imagenet_paths/dg3_<scale>_5sp/scanpath/best.pth).
  * images resized to 224x224 -- the resolution DG3 was trained at on our
    synthetic data (configs/imagenet.yaml image_size: 224).
  * flat (zero) centerbias, matching that training (centerbias.use_centerbias:
    false for the synthetic runs); DG3's finalizer adds it in log space, so a
    zero map is a true no-op.
  * one sampled scanpath per human scanpath, length-matched to it (the same
    length oracle our sampler gets), starting from the image centre.

Output: np.save dict {stem: [ (L,2) arrays in [-1,1] ]} -- exactly what
scandiff/tools/score_scanpaths.py consumes.

Usage:
  python -m scanpath.sample_scanpaths_dg3 \
      --checkpoint runs/imagenet_paths/dg3_200k_5sp/scanpath/best.pth \
      --spec /mnt/.../scandiff/data/eval/mit1003_human_full.json \
      --stimuli-dir /mnt/.../MIT1003/stimuli \
      --out /mnt/.../scandiff/data/eval/gen_mit1003_dg3_200k.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scanpath.utils.config import load_config
from scanpath.deepgaze_engine import make_scanpath_model

SIZE = 224          # DG3 synthetic-training resolution
N_HIST = 4          # DG3 uses included_fixations=[-1,-2,-3,-4]


def load_image(path: Path) -> torch.Tensor:
    """-> (3, SIZE, SIZE) float32 in [0,255], the scale DG3's Normalizer expects."""
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)          # (H, W, 3) in [0,255]
    return torch.from_numpy(arr.transpose(2, 0, 1))


@torch.no_grad()
def sample_for_image(model, image: torch.Tensor, lengths: list[int], device: str,
                     temperature: float = 1.0, generator=None) -> list[np.ndarray]:
    """
    image  : (3, SIZE, SIZE) in [0,255]
    lengths: total fixations per scanpath INCLUDING the centre start
    returns: list of (L_b, 2) arrays in [-1,1]
    """
    B = len(lengths)
    maxL = max(lengths)
    img = image.unsqueeze(0).expand(B, -1, -1, -1).to(device)
    centerbias = torch.zeros(B, SIZE, SIZE, device=device)   # flat, as in training

    # fixation history in pixel coords; row 0 = image centre
    fx = torch.full((B, maxL), float("nan"), device=device)
    fy = torch.full((B, maxL), float("nan"), device=device)
    fx[:, 0] = (SIZE - 1) / 2.0
    fy[:, 0] = (SIZE - 1) / 2.0

    for step in range(1, maxL):
        # last N_HIST fixations, most-recent first (DG3's [-1,-2,-3,-4] order)
        idx = [step - 1 - k for k in range(N_HIST)]
        x_hist = torch.stack(
            [fx[:, i] if i >= 0 else torch.full((B,), float("nan"), device=device)
             for i in idx], dim=1)
        y_hist = torch.stack(
            [fy[:, i] if i >= 0 else torch.full((B,), float("nan"), device=device)
             for i in idx], dim=1)

        log_density = model(img, centerbias, x_hist=x_hist, y_hist=y_hist,
                            durations=torch.tensor([]).to(device))
        if log_density.dim() == 4:
            log_density = log_density[:, 0]
        logits = log_density.reshape(B, -1) / temperature
        probs = torch.softmax(logits, dim=-1)
        flat = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        fy[:, step] = (flat // SIZE).float()
        fx[:, step] = (flat % SIZE).float()

    out = []
    for b, L in enumerate(lengths):
        x = fx[b, :L].cpu().numpy() / (SIZE - 1) * 2.0 - 1.0
        y = fy[b, :L].cpu().numpy() / (SIZE - 1) * 2.0 - 1.0
        out.append(np.stack([x, y], axis=1).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spec", required=True, help="human spec json (length oracle + sizes)")
    ap.add_argument("--stimuli-dir", required=True)
    ap.add_argument("--ext", default=".jpeg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="scanpath/configs/imagenet.yaml",
                    help="config the checkpoint was trained with (architecture params)")
    ap.add_argument("--stage", default="stage2", help="stage whose arch params to use")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only N images (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config, {})
    model = make_scanpath_model(cfg, args.stage, device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {args.checkpoint} | missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    if len(missing) > 20:
        raise SystemExit(f"too many missing keys ({len(missing)}) -- wrong config/stage?")
    model.eval()

    spec = json.load(open(args.spec))
    stems = sorted(spec.keys())
    if args.limit:
        stems = stems[:args.limit]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    out: dict[str, list[np.ndarray]] = {}
    n_sp = 0
    for i, stem in enumerate(stems):
        img_path = Path(args.stimuli_dir) / f"{stem}{args.ext}"
        if not img_path.exists():
            print(f"  missing image {img_path}, skipping", flush=True)
            continue
        lengths = [len(sp["X"]) for sp in spec[stem]["scanpaths"]]
        lengths = [max(2, int(l)) for l in lengths]
        image = load_image(img_path)
        out[stem] = sample_for_image(model, image, lengths, device,
                                     temperature=args.temperature, generator=gen)
        n_sp += len(lengths)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(stems)} images, {n_sp} scanpaths", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)
    print(f"saved {len(out)} images / {n_sp} scanpaths -> {args.out}")


if __name__ == "__main__":
    main()
