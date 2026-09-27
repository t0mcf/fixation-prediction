"""Score OUR model with pysaliency's reference evaluation on MIT1003.

Wraps a fixation-prediction checkpoint as a pysaliency ScanpathModel: for each
fixation, the model's conditional 64x64 density (in the [-1,1]^2 frame of the
full image) is expanded to the stimulus pixel grid by piecewise-constant mass
spreading — the unique pixel-grid density the coarse model actually asserts;
nothing is interpolated or sharpened. pysaliency then computes its own
conditional log-likelihoods, exactly as for DeepGaze III.

This is the maximally DG3-favourable comparison endpoint: on the pixel grid a
64-cell model pays the full price of its coarseness against DG3's sharp maps.

Conventions replicated from src/data/mit1003_dataset.py:
  * image resized to 224x224 (aspect-distorting) and ImageNet-normalised
  * coordinates normalised to [-1,1] by the stimulus's own width/height
  * a synthetic centre fixation (0,0) is prepended to every history
  * histories longer than max_seq_len-1 keep the most recent max_seq_len-1
    entries (training never saw longer prefixes; noted, affects only the
    tail of unusually long scanpaths)

  python -m scanpath.pysaliency_eval_ours \
      --checkpoint <fixation-prediction>/runs/ft_authsplit_lad800k_seed42/checkpoints/best_val_ll.pt \
      --fixed-split-dir .../splits/mit1003 --twosize-dir <dir> --name ours-800k
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import numpy as _np
if not hasattr(_np, "in1d"):
    _np.in1d = _np.isin  # removed in numpy 2.x; pysaliency still calls it
import pysaliency

from scanpath.deepgaze_engine import convert_fixation_trains, convert_stimuli
from scanpath.eval_mit1003_fair_dg3 import fixed_split

FIXPRED = Path("/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/fixation-prediction")
sys.path.insert(0, str(FIXPRED))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_build_model():
    spec = importlib.util.spec_from_file_location(
        "fair_ours", FIXPRED / "scripts/eval_mit1003_fair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


class OursPysaliencyWrapper(pysaliency.ScanpathModel):
    def __init__(self, model, device, id_to_path: dict, max_seq_len=16):
        super().__init__()
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        # pysaliency stimulus_id -> ORIGINAL MIT1003 image file. The image
        # must go through the dataset's own transform (torchvision Resize on
        # the original JPEG); feeding the re-encoded two-size stimulus through
        # F.interpolate cost a uniform ~0.6 bit for every model (2026-09-03).
        self.id_to_path = id_to_path
        self._token_cache: dict[str, torch.Tensor] = {}

    def _image_tokens(self, stimulus_id: str) -> torch.Tensor:
        if stimulus_id not in self._token_cache:
            from PIL import Image
            from src.data.mit1003_dataset import _img_transform
            img = Image.open(self.id_to_path[stimulus_id]).convert("RGB")
            img = _img_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self._token_cache.clear()  # one image at a time is enough
                self._token_cache[stimulus_id] = self.model.encode_image(img)
        return self._token_cache[stimulus_id]

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist,
                                attributes=None, out=None):
        stimulus_id = stimulus.stimulus_id
        stimulus = getattr(stimulus, "stimulus_data", stimulus)
        stimulus = np.ascontiguousarray(np.asarray(stimulus), dtype=np.float32)
        H, W = stimulus.shape[0], stimulus.shape[1]
        xs = np.clip(np.asarray(x_hist, np.float64) / (W - 1) * 2 - 1, -1, 1)
        ys = np.clip(np.asarray(y_hist, np.float64) / (H - 1) * 2 - 1, -1, 1)
        # pysaliency's get_mit1003_with_initial_fixation already supplies the
        # central initial fixation as x_hist[0]; it IS the centre the training
        # dataset prepends. Prepending another (0,0) here doubled the centre
        # and shifted every prefix by one (found 2026-09-03: 1.83 vs the
        # harness's 2.19 bits for scratch). Use the history as given.
        hist = list(zip(xs, ys))
        hist = hist[-(self.max_seq_len - 1):]        # most recent window
        L = len(hist)

        fixations = torch.zeros(1, self.max_seq_len, 2, device=self.device)
        fixations[0, :L] = torch.tensor(hist, dtype=torch.float32)
        # The model treats the LAST slot as the target and uses
        # prefix_len = fixations_len - 1 visible tokens. With flen = L the
        # most recent fixation was masked as padding and every prediction
        # ran one fixation short (~0.55 bit, found 2026-09-03). The target
        # slot itself stays zero — it is never read (prefix = fixations[:-1]).
        flen = torch.tensor([L + 1], device=self.device)

        tokens = self._image_tokens(stimulus_id)
        with torch.no_grad():
            log_preds = self.model(
                fixations.new_zeros(1, 3, 224, 224),  # ignored: tokens given
                fixations, flen, image_tokens=tokens).float()
        lp = log_preds[0, L - 1]                     # (64, 64), log-density

        # Piecewise-constant expansion to the pixel grid using EXACTLY the
        # cell rule of src/training/losses.py::ll_score — pixel (px, py) ->
        # cell (round(px/(W-1)*63), round(py/(H-1)*63)) — and mass-preserving
        # per cell (cell mass / number of pixels mapped to it). F.interpolate
        # 'nearest' used uniform 16-px cells from pixel 0 instead, i.e. up to
        # half a cell of misalignment against the density the harness scores
        # (uniform ~0.6-bit penalty for every model; found 2026-09-03).
        G = lp.shape[-1]
        col = torch.round(torch.arange(W, dtype=torch.float32) / (W - 1) * (G - 1)).long().clamp(0, G - 1)
        row = torch.round(torch.arange(H, dtype=torch.float32) / (H - 1) * (G - 1)).long().clamp(0, G - 1)
        counts = torch.zeros(G, G)
        counts.index_put_((row[:, None].expand(H, W).reshape(-1), col[None, :].expand(H, W).reshape(-1)),
                          torch.ones(H * W), accumulate=True)
        cell_logp = lp.cpu() - torch.log(counts.clamp_min(1.0))   # per-pixel log density
        px_log = cell_logp[row[:, None], col[None, :]]              # (H, W)
        return px_log.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fixed-split-dir", required=True)
    ap.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    ap.add_argument("--twosize-dir", required=True)
    ap.add_argument("--dataset-dir", default="/mnt/vast-nhr/projects/nim00018/datasets")
    ap.add_argument("--name", default="ours-pysaliency")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_model = _load_build_model()
    ckpt = torch.load(args.checkpoint, map_location=device)
    net = build_model(ckpt, device)

    print("Loading MIT1003 ...", flush=True)
    stim_orig, scan_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=args.dataset_dir, replace_initial_invalid_fixations=True)
    scan_twosize = convert_fixation_trains(stim_orig, scan_orig)
    stim_twosize = convert_stimuli(stim_orig, Path(args.twosize_dir))
    fix_twosize = scan_twosize[scan_twosize.lengths > 0]

    _, _, eval_idx = next(fixed_split(
        args.fixed_split_dir, args.eval_split, stim_twosize))
    eval_stim, eval_fix = pysaliency.create_subset(stim_twosize, fix_twosize, eval_idx)
    eval_fix.n = eval_fix.n.astype(int)

    # two-size stimuli and original stimuli are index-aligned; map the
    # two-size ids (what pysaliency hands the model) to the original files
    id_to_path = {sid: str(stim_orig.filenames[i])
                  for i, sid in enumerate(stim_twosize.stimulus_ids)}
    model = OursPysaliencyWrapper(net, device, id_to_path)
    lls = model.log_likelihoods(eval_stim, eval_fix, verbose=True)
    print(f"\n=== {args.name} ({args.eval_split}, {len(eval_idx)} images, "
          f"{len(lls)} fixations) ===")
    print(f"pysaliency mean LL (nat, native res): {np.mean(lls):.4f}")
    sizes = np.array(eval_stim.sizes)[eval_fix.n.astype(int)]
    uniform = -np.log(sizes[:, 0].astype(float) * sizes[:, 1].astype(float))
    print(f"bits over uniform (pixel grid)      : "
          f"{(np.mean(lls) - np.mean(uniform)) / np.log(2):.4f}")


if __name__ == "__main__":
    main()
