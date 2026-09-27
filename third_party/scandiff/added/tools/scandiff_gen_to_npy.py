"""
Convert ScanDiff's saved MIT1003 generations (pkl from src/eval.py, CUDA
tensors, coords in [0,1], padding -3) into the scoring format used by
tools/score_scanpaths.py: np.save dict {stem: [(L,2) arrays in [-1,1]]}.

This replaces the ad-hoc in-session conversion that produced
data/eval/gen_scandiff_10k_m100.npy, so ladder scoring is reproducible:

  python tools/scandiff_gen_to_npy.py \
      --gen-pkl runs_ours/train_50k/eval_mit100/generations_epoch_*/generations_MIT1003Dataset_test.pkl \
      --out data/eval/gen_scandiff_50k_m100.npy
"""
from __future__ import annotations

import argparse
import glob
import io
import pickle
from pathlib import Path

import numpy as np
import torch

PAD = -3.0


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-pkl", required=True,
                    help="generations_*_test.pkl (glob ok; newest match used)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.gen_pkl), key=lambda p: Path(p).stat().st_mtime)
    if not matches:
        raise FileNotFoundError(args.gen_pkl)
    src = matches[-1]
    print(f"loading {src}")
    with open(src, "rb") as f:
        gen = CPU_Unpickler(f).load()

    out: dict[str, list[np.ndarray]] = {}
    n_sp, n_fix = 0, 0
    for stem, rec in gen.items():
        stem = Path(stem).stem   # 'foo.jpeg' -> 'foo', matching the human json keys
        sps = rec["scanpaths"]
        sps = sps.cpu().numpy() if torch.is_tensor(sps) else np.asarray(sps)
        lst = []
        for sp in sps:                       # (T, 3) x,y,dur in [0,1], pad -3
            valid = ~np.any(sp[:, :2] <= PAD + 1e-3, axis=1)
            xy = sp[valid, :2]
            if len(xy) < 2:
                continue
            lst.append((xy * 2.0 - 1.0).astype(np.float32))   # [0,1] -> [-1,1]
            n_sp += 1
            n_fix += len(xy)
        if lst:
            out[stem] = lst

    np.save(args.out, out, allow_pickle=True)
    print(f"saved {len(out)} images / {n_sp} scanpaths / {n_fix} fixations -> {args.out}")


if __name__ == "__main__":
    main()
