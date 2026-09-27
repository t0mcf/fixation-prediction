"""
Fair MIT1003 evaluation harness (step 0 of the DG3-parity plan).

Scores a scanpath model on MIT1003 under DeepGaze III's conventions so the
numbers line up across models:

  * LL over uniform, image-averaged (DG3's average='image') and
    fixation-weighted (our historical convention), same DG3 formula.
  * Information gain over an empirical MIT1003 center bias -- neutralises the
    free bits DG3 gets from its bolt-on center-bias prior.
  * A resolution sweep (--resolutions 64 128 224): our native 64x64 masses are
    pooled up to each grid so we can read off how many bits the coarse grid
    costs before building anything.

Two modes:
  zeroshot  : one pretraining checkpoint, teacher-forced over every MIT1003
              scanpath. Every image is scored once as the held-out fold of a
              10-fold split; the center bias for each fold is fit on that
              fold's 9 training folds (leak-free). One honest zero-shot number.
  finetuned : a CV run dir with fold{0..9}/checkpoints/<ckpt-name>; each fold's
              val images scored by that fold's finetuned checkpoint, center
              bias fit on that fold's train images.

The fold partition reproduces src/training/train.py exactly:
    perm = np.random.default_rng(seed).permutation(1003)
    folds = np.array_split(perm, n_folds); fold k = validation.

Examples:
  python scripts/eval_mit1003_fair.py --mode zeroshot \
      --checkpoint runs/scale_800k_5p_ll/checkpoints/best_val_nss.pt --name 800k-zeroshot

  python scripts/eval_mit1003_fair.py --mode finetuned \
      --run-dir runs/ftcv_800k_5p --ckpt-name best_val_ll.pt --name 800k-ft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.mit1003_dataset import make_mit1003_loader, MIT1003Dataset
from src.models.dino_scanpather import ScanpathModel
from src.training.metrics import nss_score, auc_score
from src.eval.fair_metrics import (
    fit_centerbias,
    fit_centerbias_tuned,
    resize_logdensity,
    FairMetricAccumulator,
)
from src.eval.frozen_results import validate_split_files, image_results, write_results


# ---------------------------------------------------------------------------
# model construction (mirrors evaluate_dg3_compat.build_model_from_checkpoint)
# ---------------------------------------------------------------------------

def build_model(ckpt: dict, device: str) -> ScanpathModel:
    a = ckpt.get("args", {})
    model = ScanpathModel(
        visual_encoder=a.get("visual_encoder", "dinov2_base"),
        pretrained_encoder=not a.get("no_pretrained_encoder", False),
        img_size=224,
        d_model=a.get("d_model", 256),
        n_heads=a.get("n_heads", 8),
        scanpath_layers=a.get("scanpath_layers", 2),
        fusion_layers=a.get("fusion_layers", 4),
        max_seq_len=a.get("max_seq_len", 16),
        heatmap_size=a.get("heatmap_size", 64),
        dropout=a.get("dropout", 0.1),
        decoder_dropout=a.get("decoder_dropout", None),
        decoder_hidden_channels=a.get("decoder_hidden_channels", 256),
        decoder_upsample=a.get("decoder_upsample", "nearest"),

        use_visual_scanpath_features=a.get("visual_scanpath_features", False),
        use_patch_pos_embed=a.get("patch_pos_embed", False),
        use_bidirectional_fusion=a.get("bidirectional_fusion", False),
        use_fixation_query_fusion=a.get("fixation_query_fusion", False),
        use_finalizer=a.get("use_finalizer", False),
        use_centerbias=a.get("use_centerbias", False),
        use_saccadeprior=a.get("use_saccadeprior", False),
        finalizer_init_sigma=a.get("finalizer_init_sigma", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# fold partition (identical to train.py)
# ---------------------------------------------------------------------------

def fold_splits(seed: int, n_folds: int, n_images: int = 1003):
    perm = np.random.default_rng(seed).permutation(n_images)
    folds = np.array_split(perm, n_folds)
    for k in range(n_folds):
        val = sorted(folds[k].tolist())
        train = sorted(np.concatenate([folds[i] for i in range(n_folds) if i != k]).tolist())
        yield k, train, val


def fixed_split(split_dir: Path, eval_split: str):
    """Authors' fixed splits, mapped to dataset indices exactly like
    train.py's --mit-split-dir (basename match against stimuli.hdf5,
    fail-loud). Center bias is fit on train.txt; scoring runs on
    validation.txt or test.txt. Yields a single pseudo-fold."""
    validate_split_files(split_dir)
    import h5py
    from src.data.mit1003_dataset import MIT1003_ROOT
    with h5py.File(MIT1003_ROOT / "stimuli.hdf5", "r") as f:
        fnames = [fn.decode() if isinstance(fn, bytes) else fn
                  for fn in f["filenames"][:]]
    name_to_idx = {Path(fn).name.lower(): i for i, fn in enumerate(fnames)}

    def load(fname):
        wanted = [l.strip() for l in open(split_dir / fname) if l.strip()]
        missing = [w for w in wanted if Path(w).name.lower() not in name_to_idx]
        if missing:
            raise SystemExit(f"{fname}: {len(missing)} images not found, "
                             f"e.g. {missing[:3]}")
        return sorted(name_to_idx[Path(w).name.lower()] for w in wanted)

    train = load("train.txt")
    evalset = load(f"{eval_split}.txt")
    overlap = set(train) & set(evalset)
    assert not overlap, f"train/{eval_split} overlap: {len(overlap)} images"
    print(f"FIXED splits from {split_dir}: {len(train)} train (CB fit), "
          f"{len(evalset)} {eval_split} (scored)", flush=True)
    yield 0, train, evalset


def train_fold_fixations(train_images: list[int]) -> np.ndarray:
    """All target fixation coords (normalised [-1,1]) for the given train images."""
    ds = MIT1003Dataset(max_seq_len=16, min_fixations=1, image_indices=train_images)
    xy = []
    for i in range(len(ds)):
        s = ds[i]
        n = int(s["fixations_len"])
        # targets are fixations[1:n] (step t predicts fixation t+1)
        xy.append(s["fixations"][1:n].numpy())
    return np.concatenate(xy, axis=0) if xy else np.zeros((0, 2), np.float32)


def train_fold_fixations_by_image(train_images: list[int]) -> list[np.ndarray]:
    """
    All REAL fixation coords (normalised [-1,1]; the prepended synthetic centre
    fixation excluded, untruncated by max_seq_len), grouped per image for
    image-wise CB tuning. Reads dataset records directly to skip image I/O.
    """
    ds = MIT1003Dataset(max_seq_len=16, min_fixations=1, image_indices=train_images)
    by_img: dict[int, list[np.ndarray]] = {}
    for rec in ds._records:
        H, W = ds._img_shapes[rec["img_idx"]]
        n = rec["n_valid"]
        x = np.clip((rec["xs"][:n] / (W - 1)) * 2.0 - 1.0, -1.0, 1.0)
        y = np.clip((rec["ys"][:n] / (H - 1)) * 2.0 - 1.0, -1.0, 1.0)
        by_img.setdefault(rec["img_idx"], []).append(
            np.stack([x, y], axis=1).astype(np.float32))
    return [np.concatenate(v, axis=0) for v in by_img.values()]


# ---------------------------------------------------------------------------
# scoring one (model, val-images, centerbias) triple at multiple resolutions
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_fold(model, val_images, centerbias_by_res, resolutions, device,
               batch_size, num_workers):
    loader = make_mit1003_loader(
        batch_size=batch_size, num_workers=num_workers,
        max_seq_len=16, min_fixations=1, image_indices=val_images, shuffle=False,
    )
    accs = {r: FairMetricAccumulator(grid=r, centerbias=centerbias_by_res[r])
            for r in resolutions}
    # NSS/AUC are not part of FairMetricAccumulator (it owns LL/IG only).
    # Accumulate them here in BOTH conventions the accumulator uses for LL:
    #   *_fix = all fixations weighted equally  (matches how train.py validates
    #           and how the model's objective is averaged)
    #   *_img = per-image mean first, then over images (DG3 / LL_img convention)
    # so whichever the report settles on, all three metrics agree.
    per_img = {r: {} for r in resolutions}   # img_id -> [nss_sum, auc_sum, n]

    for batch in loader:
        image = batch["image"].to(device)
        fixations = batch["fixations"].to(device)
        fixations_len = batch["fixations_len"].to(device)
        img_idx = batch["img_idx"]

        B, S, _ = fixations.shape
        T = S - 1
        log_preds = model(image, fixations, fixations_len).float()  # (B,T,64,64)
        target_fix = fixations[:, 1:, :]                            # (B,T,2)

        valid = (torch.arange(T, device=device).unsqueeze(0)
                 < (fixations_len - 1).unsqueeze(1))                # (B,T)
        vflat = valid.reshape(-1)
        if not vflat.any():
            continue

        lp = log_preds.reshape(B * T, *log_preds.shape[2:])[vflat]  # (V,64,64)
        txy = target_fix.reshape(B * T, 2)[vflat].cpu()            # (V,2)
        # per-fixation image id: repeat each sample's img_idx across its T steps
        img_rep = img_idx.repeat_interleave(T)[vflat.cpu()]        # (V,)

        for r in resolutions:
            lp_r = resize_logdensity(lp, r).cpu() if r != lp.shape[-1] else lp.cpu()
            accs[r].add(lp_r, txy, img_rep)
            # nss_score/auc_score reduce internally, so score each image's
            # fixations separately to recover both aggregations.
            for iid in img_rep.unique():
                m = img_rep == iid
                n_m = int(m.sum())
                e = per_img[r].setdefault(int(iid), [0.0, 0.0, 0])
                e[0] += float(nss_score(lp_r[m], txy[m])) * n_m
                e[1] += float(auc_score(lp_r[m], txy[m])) * n_m
                e[2] += n_m

    out = {}
    for r in resolutions:
        res = accs[r].result()
        expected_fixations = sum(min(rec["n_valid"], 15) for rec in loader.dataset._records)
        if res.get("n_fixations") != expected_fixations:
            raise RuntimeError("Incomplete fixation evaluation")
        res["per_image"] = image_results(accs[r], per_img[r])
        for row in res["per_image"]:
            row["image"] = Path(loader.dataset._filenames[row["image_id"]]).name.lower()
        pi = list(per_img[r].values())
        if pi:
            tot_n = sum(e[2] for e in pi)
            res["nss_fix"] = sum(e[0] for e in pi) / tot_n
            res["auc_fix"] = sum(e[1] for e in pi) / tot_n
            res["nss_img"] = float(np.mean([e[0] / e[2] for e in pi]))
            res["auc_img"] = float(np.mean([e[1] / e[2] for e in pi]))
        out[r] = res
    return out


# ---------------------------------------------------------------------------
# aggregation across folds
# ---------------------------------------------------------------------------

def aggregate(fold_results: list[dict], resolutions):
    """fold_results[i][r] = metric dict. Returns per-resolution mean/std across folds."""
    out = {}
    for r in resolutions:
        keys = ["ll_uniform_img", "ll_uniform_fix",
                "ig_centerbias_img", "ig_centerbias_fix",
                "centerbias_ll_uniform_img",
                "nss_fix", "auc_fix", "nss_img", "auc_img"]
        agg = {}
        for key in keys:
            vals = [fr[r][key] for fr in fold_results if r in fr and key in fr[r]]
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
        agg["n_fixations"] = sum(fr[r]["n_fixations"] for fr in fold_results if r in fr)
        out[r] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "finetuned"], required=True)
    ap.add_argument("--checkpoint", help="zeroshot: single pretraining checkpoint")
    ap.add_argument("--run-dir", help="finetuned: dir with fold{k}/checkpoints/<ckpt-name>")
    ap.add_argument("--ckpt-name", default="best_val_ll.pt")
    ap.add_argument("--name", default="model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--mit-split-dir", type=str, default=None,
                    help="dir with train/validation/test.txt image lists; "
                         "replaces the CV fold loop with one run on the fixed "
                         "split. finetuned mode then takes --checkpoint "
                         "directly (a single fine-tuned checkpoint).")
    ap.add_argument("--eval-split", choices=["validation", "test"],
                    default="validation",
                    help="which fixed-split file to score (test: test day only)")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 224])
    ap.add_argument("--cb-smooth-frac", type=float, default=0.05)
    ap.add_argument("--cb-mode", choices=["legacy", "tuned"], default="tuned",
                    help="legacy: fixed-bandwidth histogram (smooth-frac, heavy "
                         "floor). tuned: per-fold cross-validated bandwidth + "
                         "eps mixture, DG3-parity (default).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--output-json", help="Exclusive full-precision fixed-split output, including per-image metrics")
    args = ap.parse_args()
    if args.output_json and Path(args.output_json).exists():
        ap.error("output JSON already exists")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  mode: {args.mode}  resolutions: {args.resolutions}", flush=True)

    zeroshot_model = None
    if args.mode == "zeroshot" or (args.mit_split_dir and args.checkpoint):
        assert args.checkpoint, "--checkpoint required for zeroshot"
        ckpt = torch.load(args.checkpoint, map_location=device)
        zeroshot_model = build_model(ckpt, device)

    if args.mit_split_dir:
        splits = fixed_split(Path(args.mit_split_dir), args.eval_split)
    else:
        splits = fold_splits(args.seed, args.n_folds)

    fold_results = []
    for k, train_images, val_images in splits:
        # center bias fit on this fold's TRAIN images, at each resolution
        if args.cb_mode == "tuned":
            cb_per_img = train_fold_fixations_by_image(train_images)
            cb_by_res = {r: fit_centerbias_tuned(cb_per_img, r, device=device,
                                                 verbose=(r == args.resolutions[0]))
                         for r in args.resolutions}
        else:
            cb_xy = train_fold_fixations(train_images)
            cb_by_res = {r: fit_centerbias(cb_xy, r, args.cb_smooth_frac, device)
                         for r in args.resolutions}

        if zeroshot_model is not None:
            model = zeroshot_model
        else:
            ckpt_path = Path(args.run_dir) / f"fold{k}" / "checkpoints" / args.ckpt_name
            if not ckpt_path.exists():
                print(f"  fold {k}: MISSING {ckpt_path}, skipping", flush=True)
                continue
            model = build_model(torch.load(ckpt_path, map_location=device), device)

        res = score_fold(model, val_images, cb_by_res, args.resolutions,
                         device, args.batch_size, args.num_workers)
        fold_results.append(res)
        r0 = args.resolutions[0]
        print(f"  fold {k}: LL_img@{r0}={res[r0]['ll_uniform_img']:.3f} "
              f"IG_cb@{r0}={res[r0]['ig_centerbias_img']:.3f} "
              f"(n_fix={res[r0]['n_fixations']})", flush=True)

        if args.mode == "finetuned" and zeroshot_model is None:
            del model
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
        def fmt(key): return f"{a[key][0]:.3f}±{a[key][1]:.3f}"
        print(f"{r:>5} {fmt('ll_uniform_img'):>16} {fmt('ll_uniform_fix'):>16} "
              f"{fmt('ig_centerbias_img'):>16} {fmt('ig_centerbias_fix'):>16} "
              f"{a['centerbias_ll_uniform_img'][0]:>10.3f} "
              f"{fmt('nss_img'):>16} {fmt('nss_fix'):>16} "
              f"{fmt('auc_img'):>16} {fmt('auc_fix'):>16}")


if __name__ == "__main__":
    main()
