"""
Build the FROZEN protocol-v2 scoring subset — the shared in-domain evaluation
harness for all three models (ours / DeepGaze III / ScanDiff).

~400 images, deterministically chosen from the shared v2 validation split
(5,000 images): stems sorted, then evenly-spaced indices. The stem list is
class-uniform because stems sort by synset. Built ONCE; the sha256 of the stem
list is the identity of the subset — keep it fixed and never rebuild.

Outputs
  data/thesis_v2/oursynth_v2_sub400/
      oursynth_v2_sub400_fixations_{train,validation,test}.json
          subset rows, subjects 0-4 only (5/image) — this drives how many
          scanpaths the evaluator GENERATES per image (~2,000 total/checkpoint)
      dinov2_base_timm_image_features -> symlink to the 50k feature memmap
          (coverage of every subset stem is verified against memmap_index)
  configs/data/{train,val,test}_datasets/oursynth_v2_sub400.yaml
  data/eval/v2_scoring_subset/
      subset_gt.json      scoring REFERENCE: all 16 AV paths per image
      subset_stems.txt    the frozen stem list (+ printed sha256)
      stimuli/<stem>.JPEG flat symlinks (for samplers that need images)

  python tools/make_v2_scoring_subset.py            # build + verify
  python tools/make_v2_scoring_subset.py --n 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SRC_JSON = ROOT / "data/thesis_v2/oursynth_v2_50k/oursynth_v2_50k_fixations_validation.json"
FEATURES = ROOT / "data/thesis_v2/oursynth_v2_50k/dinov2_base_timm_image_features"
V2_VAL_LIST = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/imagenet_av/validation_5k.txt")
IMAGENET = Path("/mnt/vast-nhr/projects/nim00018/datasets/ImageNet/train_images")
NAME = "oursynth_v2_sub400"
GEN_SUBJECTS = 5   # scanpaths generated per image at scoring time


def main():
    global NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="'test' builds the TEST-side twin (same recipe on "
                         "test_10k.txt; dataset name + output dir suffixed "
                         "_test). final evaluation only: build once, sha-pin, never "
                         "score on it before the pre-registered final pass.")
    args = ap.parse_args()
    if args.split == "test":
        src_json = SRC_JSON.with_name("oursynth_v2_50k_fixations_test.json")
        v2_list = V2_VAL_LIST.with_name("test_10k.txt")
        NAME = NAME + "_test"
        out_dir = ROOT / "data/eval/v2_scoring_subset_test"
    else:
        src_json, v2_list = SRC_JSON, V2_VAL_LIST
        out_dir = ROOT / "data/eval/v2_scoring_subset"

    rows = json.load(open(src_json))
    by_stem = defaultdict(list)
    for r in rows:
        by_stem[Path(r["name"]).stem].append(r)
    stems_all = sorted(by_stem)
    print(f"{args.split} images: {len(stems_all)}")

    # the source must be exactly the frozen v2 split list
    v2_val = {Path(line.strip()).stem for line in open(v2_list) if line.strip()}
    assert set(stems_all) == v2_val, (
        f"{args.split} JSON stems != {v2_list.name} "
        f"({len(set(stems_all) - v2_val)} extra / {len(v2_val - set(stems_all))} missing)")

    idx = np.unique(np.linspace(0, len(stems_all) - 1, args.n).astype(int))
    subset = [stems_all[i] for i in idx]
    assert len(subset) == args.n
    digest = hashlib.sha256("\n".join(subset).encode()).hexdigest()

    # feature coverage
    memmap_index = json.load(open(FEATURES / "memmap_index.json"))["index"]
    missing = [s for s in subset if s not in memmap_index]
    assert not missing, f"{len(missing)} subset stems missing from feature memmap"

    # dataset dir: subjects 0-4 drive generation count; same file for all splits
    ds_dir = ROOT / "data/thesis_v2" / NAME
    ds_dir.mkdir(parents=True, exist_ok=True)
    gen_rows = []
    for s in subset:
        rs = sorted(by_stem[s], key=lambda r: int(r.get("subject", 0)))[:GEN_SUBJECTS]
        assert len(rs) == GEN_SUBJECTS, f"{s}: only {len(rs)} paths"
        gen_rows.extend(rs)
    for split in ("train", "validation", "test"):
        out_rows = [dict(r, split=split) for r in gen_rows]
        json.dump(out_rows, open(ds_dir / f"{NAME}_fixations_{split}.json", "w"))
    feat_link = ds_dir / "dinov2_base_timm_image_features"
    if not feat_link.exists():
        feat_link.symlink_to(FEATURES)

    # hydra dataset configs in all three groups
    yaml_body = (
        f"{NAME}:\n"
        f"  _target_: src.data.components.mit1003.mit1003.MIT1003Dataset\n"
        f"  name: {NAME}\n"
        f"  root_path: ./data/thesis_v2/{NAME}\n"
        f"  task: What do you see in the image?\n"
        f"  split: {{split}}\n"
        f"  num_subjects: {GEN_SUBJECTS}\n"
        f"  time_in_ms: false\n"
        f"  use_abs_coords: true\n"
        f"  task_embeddings_file: task_embeddings.npy\n"
        f"  img_features_dir: dinov2_base_timm_image_features\n"
    )
    for group, split in (("train_datasets", "train"),
                         ("val_datasets", "valid"),
                         ("test_datasets", "test")):
        cfg = ROOT / "configs/data" / group / f"{NAME}.yaml"
        cfg.write_text(yaml_body.format(split=split))

    # scoring reference: ALL 16 AV paths per subset image, coordinates verbatim
    out = out_dir
    (out / "stimuli").mkdir(parents=True, exist_ok=True)
    spec, n_link = {}, 0
    for s in subset:
        rs = by_stem[s]
        spec[s] = {
            "size": [int(rs[0]["width"]), int(rs[0]["height"])],
            "scanpaths": [{
                "X": list(map(float, r["X"])),
                "Y": list(map(float, r["Y"])),
                "T": list(map(float, r.get("T", [100.0] * len(r["X"])))),
                "subject": int(r.get("subject", 0)),
            } for r in rs],
        }
        cls = s.split("_")[0]
        src = IMAGENET / cls / f"{s}.JPEG"
        dst = out / "stimuli" / f"{s}.JPEG"
        if src.exists():
            if not dst.exists():
                dst.symlink_to(src)
            n_link += 1
    json.dump(spec, open(out / "subset_gt.json", "w"))
    (out / "subset_stems.txt").write_text("\n".join(subset) + "\n")

    n_ref = sum(len(v["scanpaths"]) for v in spec.values())
    print(f"subset: {len(subset)} images | sha256 {digest}")
    print(f"reference paths: {n_ref} (16/image) | generation subjects: {GEN_SUBJECTS}/image")
    print(f"stimuli symlinks: {n_link}/{len(subset)}")
    print(f"wrote {ds_dir}, configs, {out}")


if __name__ == "__main__":
    main()
