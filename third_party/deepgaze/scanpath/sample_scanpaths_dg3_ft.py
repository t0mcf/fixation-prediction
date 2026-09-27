"""Sample whole scanpaths from a FINE-TUNED DeepGaze III (fixed-split MIT1003
run, stage 3 best.pth) at its native twosize resolution with the train-only
center bias, length-matched to the human scanpaths of one fixed split
(spec WITH initial fixation), starting at the image centre. Output format =
sample_for_eval.py / sample_scanpaths_dg3.py: {stem: [(L,2) in [-1,1]]}.

Model, stimuli, split and center-bias construction mirror
scanpath.eval_mit1003_fair_dg3 exactly (same functions), so the sampled
model is the one the conditional numbers come from.

  python -m scanpath.sample_scanpaths_dg3_ft --run-dir runs/dg3_mit1003_fixedsplit_scratch_rep0 \
      --fixed-split-dir <splits> --eval-split validation \
      --spec <scandiff>/data/eval/mit1003_human_initial_validation.json --out <...>/gen.npy
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pysaliency
import torch

from scanpath.deepgaze_engine import (convert_fixation_trains, convert_stimuli,
                                      make_scanpath_model)
from scanpath.eval_mit1003_fair_dg3 import fixed_split
from scanpath.train_deepgaze import TrainOnlyCenterbias
from scanpath.utils.config import load_config

N_HIST = 4


@torch.no_grad()
def sample_for_image(model, image_chw: torch.Tensor, cb_log: torch.Tensor, lengths, device,
                     temperature=1.0, generator=None):
    """image_chw (3,H,W) float [0,255]; cb_log (H,W); lengths INCLUDING the centre start."""
    _, H, W = image_chw.shape
    B, maxL = len(lengths), max(lengths)
    img = image_chw.unsqueeze(0).expand(B, -1, -1, -1).to(device)
    cb = cb_log.unsqueeze(0).expand(B, -1, -1).to(device)
    fx = torch.full((B, maxL), float("nan"), device=device)
    fy = torch.full((B, maxL), float("nan"), device=device)
    fx[:, 0] = (W - 1) / 2.0
    fy[:, 0] = (H - 1) / 2.0
    for step in range(1, maxL):
        idx = [step - 1 - k for k in range(N_HIST)]
        x_hist = torch.stack([fx[:, i] if i >= 0 else torch.full((B,), float("nan"), device=device) for i in idx], 1)
        y_hist = torch.stack([fy[:, i] if i >= 0 else torch.full((B,), float("nan"), device=device) for i in idx], 1)
        ld = model(img, cb, x_hist=x_hist, y_hist=y_hist, durations=torch.tensor([]).to(device)).float()
        if ld.dim() == 4:
            ld = ld[:, 0]
        probs = torch.softmax(ld.reshape(B, -1) / temperature, dim=-1)
        flat = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        fy[:, step] = (flat // W).float()
        fx[:, step] = (flat % W).float()
    out = []
    for b, L in enumerate(lengths):
        x = fx[b, :L].cpu().numpy() / (W - 1) * 2.0 - 1.0
        y = fy[b, :L].cpu().numpy() / (H - 1) * 2.0 - 1.0
        out.append(np.stack([x, y], 1).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", default="MIT1003_scanpath")
    ap.add_argument("--ckpt-name", default="best.pth")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "mit1003.yaml"))
    ap.add_argument("--dataset-dir", default="/mnt/vast-nhr/projects/nim00018/datasets")
    ap.add_argument("--twosize-dir", default=None)
    ap.add_argument("--fixed-split-dir", required=True)
    ap.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    ap.add_argument("--spec", required=True, help="human spec WITH initial fixation (length oracle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config, {})
    stim_orig, scan_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=args.dataset_dir, replace_initial_invalid_fixations=True)
    scan_twosize = convert_fixation_trains(stim_orig, scan_orig)
    twosize_dir = Path(args.twosize_dir) if args.twosize_dir else Path(args.run_dir) / "MIT1003_twosize"
    stim_twosize = convert_stimuli(stim_orig, twosize_dir)
    fix_twosize = scan_twosize[scan_twosize.lengths > 0]
    (k, train_idx, val_idx), = fixed_split(args.fixed_split_dir, args.eval_split, stim_twosize)
    train_stim, train_fix = pysaliency.create_subset(stim_twosize, fix_twosize, train_idx)
    train_fix.n = train_fix.n.astype(int)
    cbc = cfg.training.centerbias
    input_cb = TrainOnlyCenterbias(train_stim, train_fix, cbc.bandwidth, cbc.eps)

    ckpt = Path(args.run_dir) / args.stage / "fixedsplit" / args.ckpt_name
    model = make_scanpath_model(cfg, "stage3", device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {ckpt} | split {args.eval_split}: {len(val_idx)} images | train-only CB", flush=True)

    spec = json.load(open(args.spec))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out, n_sp = {}, 0
    val_idx = list(val_idx)[: args.limit] if args.limit else list(val_idx)
    for i, n in enumerate(val_idx):
        stem = Path(str(stim_twosize.filenames[n])).stem.lower()
        if stem not in spec:
            raise SystemExit(f"{stem} not in spec")
        image = np.array(stim_twosize.stimuli[n])                       # (H,W,3) uint8, twosize
        cb_log = torch.as_tensor(input_cb.log_density(image), dtype=torch.float32)
        image_chw = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))
        lengths = [max(2, len(sp["X"])) for sp in spec[stem]["scanpaths"]]
        out[stem] = sample_for_image(model, image_chw, cb_log, lengths, device,
                                     temperature=args.temperature, generator=gen)
        n_sp += len(lengths)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(val_idx)} images, {n_sp} scanpaths", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)
    print(f"generated {n_sp} scanpaths over {len(out)} images -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
