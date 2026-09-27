"""Filter mit1003_human_full.json to one fixed split (authors' 702/150/151
image lists in thesis_protocol/v2/splits/mit1003). Output keeps the spec
format score_scanpaths.py / sample_for_eval.py consume.
  python tools/split_mit1003_human_spec.py --split validation
  python tools/split_mit1003_human_spec.py --split test     # final evaluation only
"""
import argparse, json
from pathlib import Path
SD = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SPLITS = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003")
ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["train", "validation", "test"], required=True)
a = ap.parse_args()
stems = {Path(l.strip()).stem for l in open(SPLITS / f"{a.split}.txt") if l.strip()}
full = json.load(open(SD / "data/eval/mit1003_human_full.json"))
sub = {k: v for k, v in full.items() if k in stems}
missing = stems - set(sub)
assert not missing, f"{len(missing)} split images missing from the full spec"
out = SD / "data/eval" / f"mit1003_human_{a.split}.json"
json.dump(sub, open(out, "w"))
print(f"{a.split}: {len(sub)} images, {sum(len(v['scanpaths']) for v in sub.values())} human scanpaths -> {out}")
