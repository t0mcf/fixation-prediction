"""Cross-check the fair harness against pysaliency's reference evaluation,
for DeepGaze III only (the one model both pipelines can score natively).

Wraps a DG3 checkpoint (or the released weights) as a pysaliency
ScanpathModel and computes pysaliency's average conditional log-likelihood
over the fixed-split evaluation images at native (twosize) resolution. The
number to compare against is the fair harness's LL at its finest grid: they
will not be identical (pooling costs a sharp model some bits), but they must
agree in ordering and rough magnitude — a large disagreement means one of the
two pipelines mishandles conventions, populations, or the center bias.

  python -m scanpath.pysaliency_sanity_dg3 --released \
      --fixed-split-dir .../splits/mit1003 --twosize-dir <dir>
  python -m scanpath.pysaliency_sanity_dg3 \
      --run-dir runs/dg3_mit1003_fixedsplit_salicon_rep0 \
      --fixed-split-dir .../splits/mit1003
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import numpy as _np
if not hasattr(_np, "in1d"):
    _np.in1d = _np.isin  # removed in numpy 2.x; pysaliency still calls it
import pysaliency

from scanpath.deepgaze_engine import (
    convert_fixation_trains, convert_stimuli, make_scanpath_model)
from scanpath.utils.config import load_config
from scanpath.eval_mit1003_fair_dg3 import fixed_split

DGP = Path(__file__).resolve().parent.parent / "models" / "DeepGaze"
sys.path.insert(0, str(DGP))


class DG3PysaliencyWrapper(pysaliency.ScanpathModel):
    """Minimal conditional_log_density adapter around a DG3 torch module."""

    def __init__(self, model, centerbias_model, device, included=4):
        super().__init__()
        self.model = model
        self.cb = centerbias_model
        self.device = device
        self.included = included

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist,
                                attributes=None, out=None):
        # The KDE center bias identifies the stimulus by a hash of its
        # ORIGINAL (uint8) data — query it before any dtype conversion.
        # Passing the float32 copy raised ValueError in stimulus_ids.index
        # (run of 2026-09-01, dg3_released_pysaliency.log).
        cb = torch.as_tensor(
            self.cb.log_density(stimulus), dtype=torch.float32,
            device=self.device).unsqueeze(0)
        stimulus = getattr(stimulus, "stimulus_data", stimulus)
        stimulus = np.ascontiguousarray(np.asarray(stimulus), dtype=np.float32)
        img = torch.as_tensor(stimulus, dtype=torch.float32,
                              device=self.device)
        img = img.permute(2, 0, 1).unsqueeze(0)
        # DG3 conditions on the last `included` fixations, newest first,
        # padded with nan exactly like deepgaze_pytorch's FixationDataset.
        xh = np.full(self.included, np.nan, np.float32)
        yh = np.full(self.included, np.nan, np.float32)
        hx = np.asarray(x_hist, np.float32)[::-1][:self.included]
        hy = np.asarray(y_hist, np.float32)[::-1][:self.included]
        xh[:len(hx)] = hx
        yh[:len(hy)] = hy
        xt = torch.as_tensor(xh, device=self.device).unsqueeze(0)
        yt = torch.as_tensor(yh, device=self.device).unsqueeze(0)
        with torch.no_grad():
            ld = self.model(img, cb, x_hist=xt, y_hist=yt,
                            durations=torch.tensor([]).to(self.device))
        return ld.squeeze().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--stage", default="MIT1003_scanpath")
    ap.add_argument("--ckpt-name", default="final.pth")
    ap.add_argument("--released", action="store_true")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "mit1003.yaml"))
    ap.add_argument("--fixed-split-dir", required=True)
    ap.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    ap.add_argument("--twosize-dir", default=None)
    ap.add_argument("--dataset-dir", default="/mnt/vast-nhr/projects/nim00018/datasets")
    ap.add_argument("--name", default="dg3-pysaliency")
    args = ap.parse_args()
    if not args.released and not args.run_dir:
        ap.error("--run-dir required unless --released")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config, None)

    print("Loading MIT1003 ...", flush=True)
    stim_orig, scan_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=args.dataset_dir, replace_initial_invalid_fixations=True)
    scan_twosize = convert_fixation_trains(stim_orig, scan_orig)
    if args.twosize_dir:
        twosize_dir = Path(args.twosize_dir)
    elif args.run_dir:
        twosize_dir = Path(args.run_dir) / "MIT1003_twosize"
    else:
        ap.error("--released without --run-dir needs --twosize-dir")
    stim_twosize = convert_stimuli(stim_orig, twosize_dir)
    fix_twosize = scan_twosize[scan_twosize.lengths > 0]

    # DG3's input center bias, rebuilt exactly as in training.
    from scanpath.train_deepgaze import _make_centerbias
    dg3_cb, _, _ = _make_centerbias(cfg.training.centerbias, stim_twosize, fix_twosize)

    _, train_idx, eval_idx = next(fixed_split(
        args.fixed_split_dir, args.eval_split, stim_twosize))
    eval_stim, eval_fix = pysaliency.create_subset(stim_twosize, fix_twosize, eval_idx)
    eval_fix.n = eval_fix.n.astype(int)

    if args.released:
        from deepgaze_pytorch import DeepGazeIII as ReleasedDeepGazeIII
        net = ReleasedDeepGazeIII(pretrained=True).to(device)
        print("released weights (trained on ALL of MIT1003 -> contaminated "
              "upper-bound reference)", flush=True)
    else:
        subdir = "fixedsplit"
        ckpt = Path(args.run_dir) / args.stage / subdir / args.ckpt_name
        net = make_scanpath_model(cfg, "stage3", device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state)
    net.eval()

    model = DG3PysaliencyWrapper(net, dg3_cb, device)
    lls = model.log_likelihoods(eval_stim, eval_fix, verbose=True)
    print(f"\n=== {args.name} ({args.eval_split}, {len(eval_idx)} images, "
          f"{len(lls)} fixations) ===")
    print(f"pysaliency mean LL (nat, native res): {np.mean(lls):.4f}")
    print(f"                  in bits           : {np.mean(lls) / np.log(2):.4f}")
    # bits over uniform on the (H, W) pixel grid, pysaliency convention
    sizes = np.array(eval_stim.sizes)[eval_fix.n.astype(int)]
    uniform = -np.log(sizes[:, 0].astype(float) * sizes[:, 1].astype(float))
    print(f"bits over uniform (pixel grid)      : "
          f"{(np.mean(lls) - np.mean(uniform)) / np.log(2):.4f}")


if __name__ == "__main__":
    main()
