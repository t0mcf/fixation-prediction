"""Like scandiff_gen_to_npy.py, but cuts generated scanpath i of an image to
the length of reference scanpath i in --spec (positional pairing, as the
evaluator generates one path per GT viewer). Input must come from
eval_keep_all_positions.py (all 16 positions exported). Reports how often
the cut is longer than ScanDiff's own predicted length would have been
(unknown here → reported as share of reference lengths > 16 = 0) and the
mean reference length."""
import argparse, glob, io, json, pickle
from pathlib import Path
import numpy as np, torch

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)

ap = argparse.ArgumentParser(); ap.add_argument("--gen-pkl", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
pkl = sorted(glob.glob(a.gen_pkl))[-1]; gen = CPU_Unpickler(open(pkl, "rb")).load(); spec = {k.lower(): v for k, v in json.load(open(a.spec)).items()}   # spec keys are lower-case, generation keys are not (fix 2026-09-19)
out, n, lens, mism = {}, 0, [], 0
for stem, rec in gen.items():
    s = Path(stem).stem
    refs = spec[s.lower()]["scanpaths"]; sps = rec["scanpaths"]
    sps = sps.cpu().numpy() if torch.is_tensor(sps) else np.asarray(sps)
    if len(sps) != len(refs): mism += 1
    lst = []
    for i, sp in enumerate(sps):
        L = len(refs[min(i, len(refs) - 1)]["X"]); L = max(2, min(L, sp.shape[0]))
        xy = np.asarray(sp[:L, :2], np.float32) * 2.0 - 1.0
        lst.append(xy); lens.append(L); n += 1
    out[s] = lst
np.save(a.out, out, allow_pickle=True)
print(f"{n} scanpaths over {len(out)} images, mean cut length {np.mean(lens):.2f}, images with viewer-count mismatch {mism} -> {a.out}")
