"""Audit completed primary transfer runs and freeze selection, without inference.

Run on HLRN with the praktikum Python. This writes a NEW manifest only;
it never submits jobs or opens test images for prediction.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
DG = ROOT.parent / "scanpather"
SPLITS = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003")
sys.path.insert(0, str(ROOT))
from src.eval.frozen_results import validate_split_files


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    validate_split_files(SPLITS)
    records = []
    dependencies = set(SPLITS.glob("*.txt"))
    # Pin evaluator/model/data code, not just the top-level launcher.
    dependencies.update(ROOT.joinpath("src").rglob("*.py"))
    dependencies.update(DG.joinpath("scanpath").rglob("*.py"))
    dependencies.update(DG.joinpath("scanpath/configs").glob("*.yaml"))
    dependencies.update(DG.joinpath("models/DeepGaze/deepgaze_pytorch").rglob("*.py"))
    for name in ("eval_mit1003_fair.py", "prepare_transfer_test.py", "run_frozen_transfer.py"):
        dependencies.add(ROOT / "scripts" / name)
    for cond in ("scratch", "bv1k", "bv10k", "bv50k", "bv100k", "bv200k", "bv400k", "bv800k"):
        for seed in (42, 43):
            run = ROOT / ("runs/ft_authsplit_%s_seed%d" % (cond, seed))
            cfg = json.loads((run / "config.json").read_text())
            assert cfg["dataset"] == "mit1003" and cfg["val_batches"] == -1
            assert Path(cfg["mit_split_dir"]).resolve() == SPLITS.resolve()
            assert cfg["seed"] == seed and cfg["num_epochs"] == 30
            assert not cfg.get("ours_crop_transform", False)
            rows = [r for r in csv.DictReader((run / "log.csv").open()) if r["split"] == "val"]
            assert len(rows) == 30 and float(rows[-1]["epoch"]) == 30
            assert all(math.isfinite(float(r["ll"])) for r in rows)
            winner = max(rows, key=lambda r: float(r["ll"]))  # trainer uses strict >: first tie
            cp = run / "checkpoints/best_val_ll.pt"
            ck = load(cp)
            assert ck["epoch"] + 1 == float(winner["epoch"])
            assert ck["step"] == int(winner["step"])
            endpoint = load(run / "checkpoints/final.pt")
            assert endpoint["epoch"] == 29
            init = cfg.get("init_from")
            if cond == "scratch":
                assert init is None
            else:
                expected = ROOT / ("runs/v2_%s_5p_ll_seed42/checkpoints/best_val_ll.pt" % cond[2:])
                assert (ROOT / init).resolve() == expected.resolve()
                pre = load(expected)
                pr = [r for r in csv.DictReader((expected.parents[1] / "log.csv").open()) if r["split"] == "val"]
                pb = max(pr, key=lambda r: float(r["ll"]))
                assert pre["epoch"] + 1 == float(pb["epoch"])
                del pre
            records.append(dict(id="ours_%s_seed%d" % (cond, seed), family="ours",
                                condition=cond, replicate=seed, run=str(run), checkpoint=str(cp),
                                checkpoint_sha256=digest(cp), selected_epoch=int(ck["epoch"]) + 1,
                                validation_ll=float(winner["ll"]),
                                selection="maximum training-validation LL; fixation-weighted, native 64 grid; first tie",
                                init_path=init))
            dependencies.update((run / "config.json", run / "log.csv"))
            del ck, endpoint
            gc.collect()
            print("verified", records[-1]["id"], flush=True)
    s2 = "MIT1003_scanpath_partially_frozen_saliency_network"
    s3 = "MIT1003_scanpath"
    for cond in ("scratch", "salicon", "full1k", "full10k", "full50k", "full100k", "full200k"):
        for rep in (0, 1):
            run = DG / ("runs/dg3_mit1003_fixedsplit_%s_rep%d" % (cond, rep))
            cfg = json.loads((run / "config.json").read_text())["config"]
            assert Path(cfg["training"]["fixed_split_dir"]).resolve() == SPLITS.resolve()
            candidates = []
            for stage in (s2, s3):
                directory = run / stage / "fixedsplit"
                rows = list(csv.DictReader((directory / "log.csv").open()))
                assert all(math.isfinite(float(r["validation_LL"])) for r in rows)
                # IG differs from LL by the SAME fixed validation baseline.
                offsets = [float(r["validation_LL"]) - float(r["validation_IG"]) for r in rows]
                assert max(offsets) - min(offsets) < 1e-10
                winner = max(enumerate(rows), key=lambda pair: (float(pair[1]["validation_IG"]), pair[0]))[1]
                cp = directory / "best.pth"
                ck = load(cp)
                assert ck["step"] == int(winner["epoch"])
                final = load(directory / "final.pth")
                assert set(final) == set(ck["model"])
                assert all(torch.equal(final[k], ck["model"][k]) for k in final)
                if stage == s3:
                    assert int(rows[-1]["epoch"]) == 9
                candidates.append(dict(stage=stage, epoch=int(winner["epoch"]),
                                       ll=float(winner["validation_LL"])))
                dependencies.add(directory / "log.csv")
                del ck, final
                gc.collect()
            assert candidates[1]["ll"] >= candidates[0]["ll"]
            cp = run / s3 / "fixedsplit/best.pth"
            records.append(dict(id="dg3_%s_rep%d" % (cond, rep), family="dg3",
                                condition=cond, replicate=rep, run=str(run), checkpoint=str(cp),
                                checkpoint_sha256=digest(cp), stage=s3,
                                selected_epoch=candidates[1]["epoch"], validation_ll=candidates[1]["ll"],
                                selection="stage-3 maximum validation IG, equivalent to image-averaged LL; last tie",
                                stage_comparison=candidates,
                                truncated_pretraining=(cond == "full200k"),
                                replicate_note="replicate labels; launcher does not set distinct training RNG seeds"))
            dependencies.add(run / "config.json")
            print("verified", records[-1]["id"], flush=True)
    manifest = dict(schema_version=1, scope="Primary fixed-split MIT1003 conditional transfer, 30 runs",
                    split_dir=str(SPLITS), resolution=224, records=records,
                    selection_note="Training-validation criteria retained; final LL/NSS/AUC image-averaged at 224. No test-based selection.",
                    dependencies={str(p): digest(p) for p in sorted(dependencies)})
    with Path(args.output).open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    print("MANIFEST", args.output, "sha256", digest(args.output), flush=True)


if __name__ == "__main__":
    main()
