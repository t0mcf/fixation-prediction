"""CPU preflight for metric arithmetic, complete exports and validation targets."""
import collections
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "scanpather"))
from src.eval.fair_metrics import FairMetricAccumulator
from src.eval.frozen_results import validate_split_files, write_results
from src.training.metrics import auc_score, nss_score
from src.data.mit1003_dataset import MIT1003Dataset
from scanpath import eval_mit1003_fair_dg3 as dg
import pysaliency


def main():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("ours_fair", ROOT / "scripts/eval_mit1003_fair.py")
    ours = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ours)
    lp = torch.tensor([[.1, .2], [.3, .4]]).log()
    cb = torch.full((2, 2), .25).log()
    xy = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [-1., -1.]])
    ids = torch.tensor([0, 0, 0, 1])
    expected = (np.log2(1.6) + np.log2(.4)) / 2
    assert auc_score(lp[None], xy[:1]).item() == 1
    assert auc_score(lp[None], xy[-1:]).item() == 0
    assert auc_score(cb[None], xy[:1]).item() == .5
    assert nss_score(cb[None], xy[:1]).item() == 0

    class Model:
        def __call__(self, image, *args, **kwargs):
            return lp.expand(len(image), 1, 2, 2) if not kwargs else lp.expand(len(image), 2, 2)

    class Loader(list):
        pass

    loader = Loader([dict(image=torch.zeros(4, 3, 2, 2),
                          fixations=torch.stack([torch.zeros_like(xy), xy], 1),
                          fixations_len=torch.tensor([2] * 4), img_idx=ids)])
    loader.dataset = SimpleNamespace(_filenames=["a.jpeg", "b.jpeg"], _records=[{"n_valid": 1}] * 4)
    with patch.object(ours, "make_mit1003_loader", return_value=loader):
        a = ours.score_fold(Model(), [0, 1], {2: cb}, [2], "cpu", 4, 0)[2]
    batches = [dict(image=torch.zeros(4, 3, 2, 2), centerbias=cb.expand(4, -1, -1),
                    x_hist=torch.zeros(4, 4), y_hist=torch.zeros(4, 4), target_xy=xy, img_idx=ids)]
    with patch.object(dg, "ImageDatasetSampler", return_value=None), patch.object(torch.utils.data, "DataLoader", return_value=batches):
        b = dg.score_fold(Model(), [None] * 4, [2], "cpu", 4, 0, {2: cb})[2]
    for key in ("ll_uniform_img", "ll_uniform_fix", "nss_img", "nss_fix", "auc_img", "auc_fix", "n_fixations", "n_images"):
        assert abs(a[key] - b[key]) < 1e-6, key
    assert abs(a["ll_uniform_img"] - expected) < 1e-6
    assert a["auc_img"] == .5 and a["auc_fix"] == .75
    print("PASS analytic metrics, both accumulation loops, unequal image weighting")

    splits = Path("/mnt/vast-nhr/projects/nim00018/tom/thesis_protocol/v2/splits/mit1003")
    groups = validate_split_files(splits)
    _, train, val = next(ours.fixed_split(splits, "validation"))
    ds = MIT1003Dataset(image_indices=val, min_fixations=1)
    targets = collections.Counter()
    for rec in ds._records:
        h, w = ds._img_shapes[rec["img_idx"]]
        n = rec["n_valid"]
        coords = np.stack([np.clip(rec["xs"][:n] / (w - 1) * 2 - 1, -1, 1),
                           np.clip(rec["ys"][:n] / (h - 1) * 2 - 1, -1, 1)], 1).astype(np.float32)
        for x, y in coords:
            targets[(rec["img_idx"], float(x), float(y))] += 1
    stimuli, scans = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location="/mnt/vast-nhr/projects/nim00018/datasets", replace_initial_invalid_fixations=True)
    fix = scans[scans.lengths > 0]
    dg_targets = collections.Counter()
    valset = set(val)
    for x, y, n in zip(fix.x, fix.y, fix.n):
        n = int(n)
        if n not in valset:
            continue
        h, w = stimuli.shapes[n][:2]
        coord = np.clip(np.array([x / (w - 1) * 2 - 1, y / (h - 1) * 2 - 1]), -1, 1).astype(np.float32)
        dg_targets[(n, float(coord[0]), float(coord[1]))] += 1
    assert targets == dg_targets and sum(targets.values()) == 15728
    print("PASS original-coordinate validation targets identical: 150 images, 15728 fixations")

    # Full-image export, precision preservation, and fail-closed output checks.
    rows = [dict(image=name, image_id=i, n_fixations=1, ll=.1234567890123, ig=0., nss=0., auc=.5)
            for i, name in enumerate(sorted(groups["validation"]))]
    args = SimpleNamespace(mit_split_dir=str(splits), eval_split="validation")
    with tempfile.TemporaryDirectory(prefix="thesis-preflight-") as tmp:
        out = Path(tmp) / "result.json"
        write_results(out, args, [{224: dict(per_image=rows, n_fixations=150)}])
        assert json.loads(out.read_text())["results"]["224"]["per_image"][0]["ll"] == .1234567890123
        try:
            write_results(out, args, [{224: dict(per_image=rows, n_fixations=150)}])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Overwrite accepted")
        try:
            write_results(Path(tmp) / "partial.json", args, [{224: dict(per_image=rows[:-1], n_fixations=149)}])
        except ValueError:
            pass
        else:
            raise AssertionError("Partial output accepted")
    print("PASS full-precision export, overwrite rejection, partial-set rejection")
    print("CPU PREFLIGHT PASSED; full GPU validation remains required")


if __name__ == "__main__":
    main()
