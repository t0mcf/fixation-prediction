"""MIT1003 human scanpaths WITH the initial (central) fixation, per fixed split,
in the spec format score_scanpaths.py / sample_for_eval.py consume.

Why a second spec (2026-09-14): ScanDiff's own evaluator scores against
MIT1003 records that include the initial fixation (paper_reproduction json:
first fixation ≈ (499.5, 377.7) on 1024x768, length L+1), whereas
mit1003_human_full.json was built from pysaliency's plain get_mit1003 and
drops it. Our and DG3's samplers start every generated path at the image
centre (row 0), so scoring them against a reference that also starts at
the centre — and length-matching to L+1 — puts all three families on the
same convention. Source: pysaliency get_mit1003_with_initial_fixation
(replace_initial_invalid_fixations=True), the same call scanpather's fair
evaluator uses. No minimum length (paths with fewer than three fixations
are padded by the metric code, as in ScanDiff's evaluator).

  python tools/extract_mit1003_human_initial.py --split validation
  python tools/extract_mit1003_human_initial.py --split test   # final evaluation only
"""
import argparse, json
from pathlib import Path
import numpy as np
import pysaliency

MIT = "/mnt/vast-nhr/projects/nim00018/datasets"
SPLITS = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003")
OUT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/eval")

ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["train", "validation", "test"], required=True)
a = ap.parse_args()
stems = {Path(l.strip()).stem.lower() for l in open(SPLITS / f"{a.split}.txt") if l.strip()}
stim, fix = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
    location=MIT, replace_initial_invalid_fixations=True)
data = {}
for t in range(len(fix.train_xs)):
    n = int(fix.train_ns[t]); stem = Path(str(stim.filenames[n])).stem.lower()
    if stem not in stems:
        continue
    h, w = stim.shapes[n][:2]
    xs = np.asarray(fix.train_xs[t], float); ys = np.asarray(fix.train_ys[t], float)
    ts = np.asarray(fix.train_ts[t], float)
    # pysaliency pads train_* arrays with NaN to the longest train: strip it
    valid = ~np.isnan(xs)
    xs, ys, ts = xs[valid], ys[valid], ts[valid]
    if len(xs) < 2:            # initial fixation only -> nothing to predict
        continue
    d = data.setdefault(stem, {"size": [int(w), int(h)], "scanpaths": []})
    d["scanpaths"].append({"X": xs.tolist(), "Y": ys.tolist(),
                           "T": (np.diff(ts, append=ts[-1] + 0.25) * 1000.0).tolist(),
                           "subject": int(fix.train_subjects[t])})
missing = stems - set(data)
assert not missing, f"{len(missing)} split images have no scanpaths"
out = OUT / f"mit1003_human_initial_{a.split}.json"
json.dump(data, open(out, "w"))
n_sp = sum(len(v["scanpaths"]) for v in data.values())
lens = [len(s["X"]) for v in data.values() for s in v["scanpaths"]]
print(f"{a.split}: {len(data)} images, {n_sp} scanpaths (incl. initial fixation), "
      f"mean length {np.mean(lens):.2f}, min {min(lens)} -> {out}")
