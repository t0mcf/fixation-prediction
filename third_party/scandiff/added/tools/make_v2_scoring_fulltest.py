"""Full-test counterpart of the frozen 400-image whole-scanpath test subset
(2026-09-21, appendix control: how well does the 400-image subset represent
the 10,000 test images?). Same file layout as data/eval/v2_scoring_subset_test,
for ALL images of test_10k, written to data/eval/v2_scoring_full_test — the
frozen subset directory and the ScanDiff dataset configs are NOT touched.
Only our model and DeepGaze III are evaluated here (ScanDiff generation on
10,000 images is too expensive). Pure evaluation of frozen checkpoints."""
import hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SRC = ROOT / "data/thesis_v2/oursynth_v2_50k/oursynth_v2_50k_fixations_test.json"
LIST = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av/test_10k.txt")
IMAGENET = Path("/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images")
OUT = ROOT / "data/eval/v2_scoring_full_test"

rows = json.load(open(SRC))
by_stem = defaultdict(list)
for r in rows:
    by_stem[Path(r["name"]).stem].append(r)
stems = sorted(by_stem)
want = {Path(l.strip()).stem for l in open(LIST) if l.strip()}
assert set(stems) == want, (len(set(stems) - want), len(want - set(stems)))
assert all(len(by_stem[s]) == 16 for s in stems), "not 16 paths for every image"

(OUT / "stimuli").mkdir(parents=True, exist_ok=True)
spec, n_link = {}, 0
for s in stems:
    rs = sorted(by_stem[s], key=lambda r: int(r.get("subject", 0)))
    spec[s] = {"size": [int(rs[0]["width"]), int(rs[0]["height"])],
               "scanpaths": [{"X": list(map(float, r["X"])), "Y": list(map(float, r["Y"])),
                              "T": list(map(float, r.get("T", [100.0] * len(r["X"])))),
                              "subject": int(r.get("subject", 0))} for r in rs]}
    src = IMAGENET / s.split("_")[0] / f"{s}.JPEG"; dst = OUT / "stimuli" / f"{s}.JPEG"
    assert src.exists(), src
    if not dst.exists():
        dst.symlink_to(src)
    n_link += 1
json.dump(spec, open(OUT / "subset_gt.json", "w"))
json.dump({s: {"size": v["size"], "scanpaths": v["scanpaths"][:5]} for s, v in spec.items()},
          open(OUT / "subset_gt_5ref.json", "w"))
(OUT / "subset_stems.txt").write_text("\n".join(stems) + "\n")
sub = {l.strip() for l in open(ROOT / "data/eval/v2_scoring_subset_test/subset_stems.txt") if l.strip()}
print(f"images {len(stems)} | 16 refs each | 5 generation paths (subjects 0-4) | links {n_link}")
print(f"frozen 400-image subset contained: {sub <= set(stems)} | sha256 {hashlib.sha256(chr(10).join(stems).encode()).hexdigest()[:16]}")
