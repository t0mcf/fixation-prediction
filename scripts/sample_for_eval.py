"""
Generate length-matched scanpaths from our model for a set of stimuli, for
sequence-metric comparison against human scanpaths (ScanDiff-style eval).

Reads a human spec JSON (from scandiff/tools/extract_mit1003_human.py):
  { "<stem>": {"size":[W,H], "scanpaths":[{"X","Y","T"}, ...]}, ... }
For each stimulus we load <stimuli_dir>/<stem>.jpeg and sample ONE scanpath per
human scanpath, matched to that human scanpath's length (our model has no
termination head, so lengths must be supplied).

Output: a pickled dict {stem: [ (L,2) arrays in [-1,1], ... ]} saved via np.save.
The scorer converts [-1,1] -> pixels with each image's (W,H).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.sample_scanpaths import build_model, sample_scanpaths, load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spec", required=True, help="human spec json")
    ap.add_argument("--stimuli-dir", required=True)
    ap.add_argument("--ext", default=".jpeg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(ckpt, args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    stim_dir = Path(args.stimuli_dir)

    generated: dict[str, list] = {}
    for n, (stem, entry) in enumerate(spec.items()):
        lengths = [len(sp["X"]) for sp in entry["scanpaths"]]
        img_path = stim_dir / f"{stem}{args.ext}"
        if not img_path.exists():
            print(f"  WARN missing image {img_path}, skipping"); continue
        img = load_image(str(img_path)).to(args.device)
        batch = img.unsqueeze(0).expand(len(lengths), -1, -1, -1).contiguous()
        sps = sample_scanpaths(model, batch, lengths, args.device,
                               temperature=args.temperature, jitter=args.jitter, generator=gen)
        generated[stem] = sps
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(spec)} images", flush=True)

    np.save(args.out, generated, allow_pickle=True)
    total = sum(len(v) for v in generated.values())
    print(f"generated {total} scanpaths over {len(generated)} images -> {args.out}")


if __name__ == "__main__":
    main()
