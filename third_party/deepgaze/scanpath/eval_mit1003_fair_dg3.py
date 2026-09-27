"""
Score DeepGaze III MIT1003 CV checkpoints through fixation-prediction's fair
harness (FairMetricAccumulator), so DG3 and our model share ONE evaluator:
identical LL convention, identical grids, identical fold-fit tuned center-bias
reference, identical seed-42 image folds.

DG3's native log-density (twosize stimulus resolution, aspect-preserving) is
mass-pooled onto the shared normalized square grid before scoring, exactly
mirroring how our model's 64x64 maps are pooled up/down by resize_logdensity.

Expects a run dir produced by train_deepgaze.py mit1003 mode with
fold_split_seed set (cross-model fold parity), e.g.:
  <run-dir>/MIT1003_scanpath_partially_frozen_saliency_network/crossval-10-<k>/final.pth

Usage:
  python -m scanpath.eval_mit1003_fair_dg3 \
      --run-dir runs/dg3_mit1003_ourfolds_from100k --name dg3-100k-ft \
      --stage MIT1003_scanpath_partially_frozen_saliency_network \
      --resolutions 64 128 224
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import pysaliency

from scanpath.deepgaze_engine import (
    convert_fixation_trains,
    convert_stimuli,
    make_scanpath_model,
)
from scanpath.utils.config import load_config

sys.path.insert(0, "/mnt/vast-nhr/home/tomcosmo.fischer/u27846/repos/fixation-prediction")
from src.training.metrics import nss_score, auc_score  # noqa: E402
from src.eval.frozen_results import validate_split_files, image_results, write_results
from src.eval.fair_metrics import (  # noqa: E402
    FairMetricAccumulator,
    fit_centerbias_tuned,
)

DGP = Path(__file__).resolve().parent.parent / "models" / "DeepGaze"
sys.path.insert(0, str(DGP))
from deepgaze_pytorch.data import FixationDataset  # noqa: E402
from deepgaze_pytorch.data import ImageDatasetSampler  # noqa: E402


# ---------------------------------------------------------------------------
# fold split — byte-identical to fixation-prediction train.py / fair eval
# ---------------------------------------------------------------------------

def fold_splits(seed: int, n_folds: int, n_images: int):
    perm = np.random.default_rng(seed).permutation(n_images)
    folds = np.array_split(perm, n_folds)
    for k in range(n_folds):
        val = sorted(folds[k].tolist())
        train = sorted(np.concatenate(
            [folds[i] for i in range(n_folds) if i != k]).tolist())
        yield k, train, val


def fixed_split(split_dir, eval_split, stimuli):
    """Authors' fixed splits mapped to stimulus indices by basename
    (fail-loud, same convention as train_deepgaze's fixed_split_dir).
    Yields one pseudo-fold: (0, train indices for CB fitting, eval indices)."""
    split_dir = Path(split_dir)
    validate_split_files(split_dir)
    name_to_idx = {Path(fn).name.lower(): i
                   for i, fn in enumerate(stimuli.filenames)}

    def load(fname):
        wanted = [l.strip() for l in open(split_dir / fname) if l.strip()]
        missing = [w for w in wanted if Path(w).name.lower() not in name_to_idx]
        if missing:
            raise SystemExit(f"{fname}: {len(missing)} images not found, "
                             f"e.g. {missing[:3]}")
        return sorted(name_to_idx[Path(w).name.lower()] for w in wanted)

    train = load("train.txt")
    evalset = load(f"{eval_split}.txt")
    assert not set(train) & set(evalset), "train/eval overlap"
    print(f"FIXED splits from {split_dir}: {len(train)} train (CB fit), "
          f"{len(evalset)} {eval_split} (scored)", flush=True)
    yield 0, train, evalset


def fixations_by_image_normalized(stimuli, fixations) -> list[np.ndarray]:
    """Per-image (n_i, 2) arrays of [-1,1] (x, y), for CB tuning."""
    sizes = list(stimuli.sizes)  # (H, W) per stimulus, subset-local indices
    by_img: dict[int, list] = {}
    for i in range(len(fixations.x)):
        n = int(fixations.n[i])
        H, W = sizes[n]
        x = np.clip(fixations.x[i] / (W - 1) * 2.0 - 1.0, -1.0, 1.0)
        y = np.clip(fixations.y[i] / (H - 1) * 2.0 - 1.0, -1.0, 1.0)
        by_img.setdefault(n, []).append((x, y))
    return [np.asarray(v, dtype=np.float32) for v in by_img.values()]


# ---------------------------------------------------------------------------
# density pooling: native (H, W) log-density -> (g, g) normalized-square grid
# ---------------------------------------------------------------------------

def pool_logdensity_to_grid(log_density: torch.Tensor, grid: int) -> torch.Tensor:
    """(N, H, W) native log-density -> (N, grid, grid) log-density, by mass."""
    prob = log_density.exp().unsqueeze(1)                       # (N,1,H,W)
    pooled = F.adaptive_avg_pool2d(prob, (grid, grid)).squeeze(1)
    pooled = pooled.clamp_min(0)
    pooled = pooled / pooled.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return torch.log(pooled.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# wrap FixationDataset so each sample keeps target pixel coords + image id
# ---------------------------------------------------------------------------

class FairSampleDataset(torch.utils.data.Dataset):
    """FixationDataset minus the mask transform: keeps x, y, adds n and (H, W)."""

    def __init__(self, base: FixationDataset, original_target_xy=None):
        self.base = base
        self.original_target_xy = original_target_xy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]  # no transform: has image, x, y, x_hist, y_hist, centerbias
        n = int(self.base.fixations.n[i])
        img = torch.as_tensor(d["image"], dtype=torch.float32)
        H, W = img.shape[1], img.shape[2]
        tx = float(d["x"][0]) / (W - 1) * 2.0 - 1.0
        ty = float(d["y"][0]) / (H - 1) * 2.0 - 1.0
        if self.original_target_xy is not None:
            tx, ty = self.original_target_xy[i]
        return {
            "image": img,
            "centerbias": torch.as_tensor(d["centerbias"], dtype=torch.float32),
            "x_hist": torch.as_tensor(d["x_hist"], dtype=torch.float32),
            "y_hist": torch.as_tensor(d["y_hist"], dtype=torch.float32),
            "target_xy": torch.tensor([tx, ty], dtype=torch.float32),
            "img_idx": torch.tensor(n, dtype=torch.long),
        }

    # ImageDatasetSampler groups same-shape stimuli into batches via this hook
    def get_shapes(self):
        return self.base.get_shapes()


@torch.no_grad()
def score_fold(model, dataset, resolutions, device, batch_size, num_workers,
               centerbias_by_res, limit_batches=None, original_target_xy=None):
    wrapped = FairSampleDataset(dataset, original_target_xy)
    loader = torch.utils.data.DataLoader(
        wrapped,
        batch_sampler=ImageDatasetSampler(wrapped, batch_size=batch_size, shuffle=False),
        num_workers=num_workers,
    )
    # NSS/AUC in both aggregations, mirroring ll_uniform_{fix,img}; see the
    # matching comment in fixation-prediction/scripts/eval_mit1003_fair.py.
    per_img = {r: {} for r in resolutions}   # img_id -> [nss_sum, auc_sum, n]
    accs = {r: FairMetricAccumulator(grid=r, centerbias=centerbias_by_res[r])
            for r in resolutions}

    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        image = batch["image"].to(device)
        centerbias = batch["centerbias"].to(device)
        x_hist = batch["x_hist"].to(device)
        y_hist = batch["y_hist"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            log_density = model(image, centerbias,
                                x_hist=x_hist, y_hist=y_hist,
                                durations=torch.tensor([]).to(device)).float()
        if log_density.dim() == 4:
            log_density = log_density[:, 0]

        txy = batch["target_xy"]
        idx = batch["img_idx"]
        for r in resolutions:
            lp_r = pool_logdensity_to_grid(log_density, r).cpu()
            accs[r].add(lp_r, txy, idx)
            for iid in idx.unique():
                m = idx == iid
                n_m = int(m.sum())
                e = per_img[r].setdefault(int(iid), [0.0, 0.0, 0])
                e[0] += float(nss_score(lp_r[m], txy[m])) * n_m
                e[1] += float(auc_score(lp_r[m], txy[m])) * n_m
                e[2] += n_m

    out = {}
    for r in resolutions:
        res = accs[r].result()
        if limit_batches is None and res.get("n_fixations") != len(dataset):
            raise RuntimeError("Incomplete fixation evaluation")
        res["per_image"] = image_results(accs[r], per_img[r])
        pi = list(per_img[r].values())
        if pi:
            tot_n = sum(e[2] for e in pi)
            res["nss_fix"] = sum(e[0] for e in pi) / tot_n
            res["auc_fix"] = sum(e[1] for e in pi) / tot_n
            res["nss_img"] = float(np.mean([e[0] / e[2] for e in pi]))
            res["auc_img"] = float(np.mean([e[1] / e[2] for e in pi]))
        out[r] = res
    return out


def aggregate(fold_results, resolutions):
    out = {}
    for r in resolutions:
        keys = ["nss_img", "nss_fix", "auc_img", "auc_fix",
                "ll_uniform_img", "ll_uniform_fix",
                "ig_centerbias_img", "ig_centerbias_fix",
                "centerbias_ll_uniform_img"]
        agg = {}
        for key in keys:
            vals = [fr[r][key] for fr in fold_results if r in fr and key in fr[r]]
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
        agg["n_fixations"] = sum(fr[r]["n_fixations"] for fr in fold_results if r in fr)
        out[r] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="train_deepgaze mit1003 out_dir (must have been trained "
                         "with fold_split_seed matching --seed)")
    ap.add_argument("--released", action="store_true",
                    help="score the authors' released DeepGaze III weights "
                         "(10-component mixture, model-zoo download) instead "
                         "of local checkpoints. Reference only: the released "
                         "model was trained on all of MIT1003, so on our "
                         "held-out splits it is potentially train-contaminated "
                         "— an upper bound, never a fair competitor.")
    ap.add_argument("--stage", default="MIT1003_scanpath_partially_frozen_saliency_network",
                    help="stage subdir holding crossval-<n>-<k>/<ckpt>")
    ap.add_argument("--ckpt-name", default="final.pth")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "mit1003.yaml"))
    ap.add_argument("--backbone", default=None,
                    help="override model.backbone (densenet|dinov2); must match "
                         "what the checkpoints were trained with")
    ap.add_argument("--name", default="dg3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 224])
    ap.add_argument("--dataset-dir", default="/mnt/vast-nhr/projects/nim00018/datasets")
    ap.add_argument("--twosize-dir", default=None,
                    help="existing MIT1003_twosize dir (default: <run-dir>/MIT1003_twosize)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--fixed-split-dir", type=str, default=None,
                    help="dir with train/validation/test.txt image lists; "
                         "replaces the CV fold loop with one run on that "
                         "split (checkpoint from <stage>/fixedsplit/).")
    ap.add_argument("--eval-split", choices=["validation", "test"],
                    default="validation",
                    help="fixed-split file to score (test: test day only)")
    ap.add_argument("--limit-folds", type=int, default=None,
                    help="score only the first N folds (smoke tests)")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="score only the first N batches per fold (smoke tests)")
    ap.add_argument("--output-json", help="Exclusive full-precision fixed-split output, including per-image metrics")
    args = ap.parse_args()
    if args.output_json and (args.limit_batches is not None or args.limit_folds is not None):
        ap.error("JSON export requires a complete evaluation")
    if args.output_json and Path(args.output_json).exists():
        ap.error("output JSON already exists")
    if not args.released and not args.run_dir:
        ap.error("--run-dir required unless --released")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config, {"backbone": args.backbone} if args.backbone else {})

    released_model = None
    if args.released:
        from deepgaze_pytorch import DeepGazeIII as ReleasedDeepGazeIII
        released_model = ReleasedDeepGazeIII(pretrained=True).to(device)
        released_model.eval()
        print("using RELEASED DeepGaze III weights (10-component mixture; "
              "trained on all of MIT1003 -> contaminated upper-bound "
              "reference on held-out splits)", flush=True)
    print(f"device: {device}  run-dir: {args.run_dir}  resolutions: {args.resolutions}",
          flush=True)

    print("Loading MIT1003 …", flush=True)
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
    fix_orig = scan_orig[scan_orig.lengths > 0]
    n_images = len(stim_twosize.filenames)

    # flat centerbias as DG3 *input* is wrong — the model was trained with the
    # KDE baseline; rebuild it identically. Released weights and CV runs:
    # ALL fixations, LOO per image (authors' formulation). Fixed-split
    # fine-tunes (2026-09-03): TRAIN-ONLY prior, rebuilt per split below.
    from scanpath.train_deepgaze import _make_centerbias, TrainOnlyCenterbias
    dg3_cb = None
    if not args.fixed_split_dir or released_model is not None:
        dg3_cb, _, _ = _make_centerbias(cfg.training.centerbias, stim_twosize, fix_twosize)

    if args.fixed_split_dir:
        splits = fixed_split(args.fixed_split_dir, args.eval_split, stim_twosize)
    else:
        splits = fold_splits(args.seed, args.n_folds, n_images)

    fold_results = []
    for k, train_idx, val_idx in splits:
        if args.limit_folds is not None and k >= args.limit_folds:
            break
        if released_model is None:
            subdir = ("fixedsplit" if args.fixed_split_dir
                      else f"crossval-{args.n_folds}-{k}")
            ckpt = Path(args.run_dir) / args.stage / subdir / args.ckpt_name
            if not ckpt.exists():
                if args.fixed_split_dir:
                    raise FileNotFoundError(ckpt)
                alt = ckpt.with_name("best.pth")
                if alt.exists():
                    ckpt = alt
                else:
                    print(f"  fold {k}: MISSING {ckpt}, skipping", flush=True)
                    continue

        train_stim, train_fix = pysaliency.create_subset(stim_twosize, fix_twosize, train_idx)
        val_stim, val_fix = pysaliency.create_subset(stim_twosize, fix_twosize, val_idx)
        val_fix.n = val_fix.n.astype(int)

        # The shared scoring grid uses original-image endpoint-normalised
        # coordinates, exactly as the ours evaluator. Scaling via new_W/W
        # and subsequently dividing by new_W-1 moves some targets by a cell.
        # Keep DG3's native image/history/input-CB representation unchanged.
        ref_train_stim, ref_train_fix = pysaliency.create_subset(stim_orig, fix_orig, train_idx)
        ref_val_stim, ref_val_fix = pysaliency.create_subset(stim_orig, fix_orig, val_idx)
        if not np.array_equal(ref_val_fix.n, val_fix.n) or len(ref_val_fix.x) != len(val_fix.x):
            raise RuntimeError("Original/native fixation ordering differs")
        original_target_xy = np.asarray([
            (x / (ref_val_stim.shapes[int(n)][1] - 1) * 2 - 1,
             y / (ref_val_stim.shapes[int(n)][0] - 1) * 2 - 1)
            for x, y, n in zip(ref_val_fix.x, ref_val_fix.y, ref_val_fix.n)
        ], dtype=np.float32).clip(-1, 1)

        # IG reference: tuned CB fit on this fold's TRAIN fixations — the same
        # object eval_mit1003_fair.py fits for our model on the same folds.
        cb_per_img = fixations_by_image_normalized(ref_train_stim, ref_train_fix)
        cb_by_res = {r: fit_centerbias_tuned(cb_per_img, r,
                                             verbose=(r == args.resolutions[0]))
                     for r in args.resolutions}

        if released_model is not None:
            model = released_model
        else:
            model = make_scanpath_model(cfg, "stage3", device)
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            model.eval()

        input_cb = dg3_cb
        if args.fixed_split_dir and released_model is None:
            cbc = cfg.training.centerbias
            train_fix.n = train_fix.n.astype(int)
            input_cb = TrainOnlyCenterbias(train_stim, train_fix, cbc.bandwidth, cbc.eps)
            print("  DG3 input center bias: TRAIN-ONLY prior (fixed-split fine-tune)", flush=True)
        dataset = FixationDataset(
            val_stim, val_fix, input_cb,
            included_fixations=[-1, -2, -3, -4],
            allow_missing_fixations=True,
            average="image",
        )
        res = score_fold(model, dataset, args.resolutions, device,
                         args.batch_size, args.num_workers, cb_by_res,
                         limit_batches=args.limit_batches,
                         original_target_xy=original_target_xy)
        for values in res.values():
            for row in values["per_image"]:
                row["image"] = Path(val_stim.filenames[row["image_id"]]).name.lower()
        fold_results.append(res)
        r0 = args.resolutions[0]
        print(f"  fold {k}: LL_img@{r0}={res[r0]['ll_uniform_img']:.3f} "
              f"IG_cb@{r0}={res[r0]['ig_centerbias_img']:.3f} "
              f"(n_fix={res[r0]['n_fixations']})", flush=True)

        if released_model is None:
            del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not fold_results:
        raise RuntimeError("no folds scored")

    if args.output_json:
        write_results(args.output_json, args, fold_results)

    agg = aggregate(fold_results, args.resolutions)
    print(f"\n=== {args.name}  ({len(fold_results)} folds) ===")
    print(f"{'res':>5} {'LL_img':>16} {'LL_fix':>16} {'IG_cb_img':>16} "
          f"{'IG_cb_fix':>16} {'CB_LL_img':>10} "
          f"{'NSS_img':>16} {'NSS_fix':>16} {'AUC_img':>16} {'AUC_fix':>16}")
    for r in args.resolutions:
        a = agg[r]
        def fmt(key):
            return f"{a[key][0]:.3f}±{a[key][1]:.3f}"
        print(f"{r:>5} {fmt('ll_uniform_img'):>16} {fmt('ll_uniform_fix'):>16} "
              f"{fmt('ig_centerbias_img'):>16} {fmt('ig_centerbias_fix'):>16} "
              f"{a['centerbias_ll_uniform_img'][0]:>10.3f} "
              f"{fmt('nss_img'):>16} {fmt('nss_fix'):>16} "
              f"{fmt('auc_img'):>16} {fmt('auc_fix'):>16}")


if __name__ == "__main__":
    main()
