"""Aggregate all protocol-v2 results into docs/report_tables/*.csv.

Every thesis figure reads from these CSVs; this script is the single place
where run outputs (eval logs, score files) become citable numbers. Re-run it
whenever new results land.

  python scripts/aggregate_v2_results.py
"""
from __future__ import annotations

import csv
import numpy as np
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RT = REPO / "docs/report_tables"
SD = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
SCANPATHER = Path.home() / "repos/scanpather"

# --split test (final evaluation only) reads the *_test twins of every input location
# and writes *_test.csv next to the validation tables, so figures switch
# splits by file name and nothing is ever overwritten.
SPLIT = "validation"
_SFX = {"validation": "", "test": "_test"}


def sfx() -> str:
    return _SFX[SPLIT]

SIZES = ["1k", "10k", "50k", "100k", "200k", "400k", "800k"]
N_IMAGES = {"1k": 1000, "10k": 10000, "50k": 50000, "100k": 100000,
            "200k": 200000, "400k": 400000, "800k": 800000}


def parse_eval_log(path: Path) -> dict | None:
    """Summary block of scripts/evaluate_checkpoint.py output."""
    if not path.exists():
        return None
    text = path.read_text()
    out = {}
    for key in ("kl", "ll", "nss", "auc"):
        m = re.search(rf"^{key}:\s+(-?[0-9.]+)", text, re.M)
        if not m:
            return None
        out[key] = float(m.group(1))
    return out


SYNTH_FAIR = REPO / "runs/synthetic_fair"


def _synth_fair_rows(prefix: str):
    """metrics.json files written by scripts/eval_synthetic_fair.py
    (2026-09-11) for one model family, on the current split."""
    d = SYNTH_FAIR / ("test" if SPLIT == "test" else "validation")
    out = []
    for f in sorted(d.glob(f"{prefix}_*/metrics.json")):
        m = json.loads(f.read_text())
        r = m["results"]
        out.append((f.parent.name, r, m["provenance"]))
    return out


def _indomain_cb() -> dict[str, dict[int, float]]:
    """image name -> {grid: centre-bias LL (bits over uniform)} for the current
    split, from scripts/indomain_centerbias_ig.py; empty dict if not run yet."""
    f = RT / f"indomain_centerbias_{'test' if SPLIT == 'test' else 'validation'}.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        for r in csv.DictReader(fh):
            out[r["image"]] = {g: float(r[f"cb_ll_{g}"]) for g in (64, 224) if f"cb_ll_{g}" in r}
    return out


def _ig(r_res: dict, cb: dict, grid: int):
    """image-averaged IG = mean over images of (LL_img − CB_img); None if no CB."""
    if not cb:
        return None
    vals = [p["ll"] - cb[p["image"]][grid] for p in r_res["per_image"] if p["image"] in cb]
    if len(vals) != r_res["n_images"]:
        raise SystemExit(f"centre-bias table covers {len(vals)} of {r_res['n_images']} images")
    return round(float(np.mean(vals)), 4)


def ours_indomain():
    """In-domain ours rows from the shared evaluator (explicit validation_5k
    list, 16 paths/image, coverage-asserted). Columns keep the names the
    figure/table scripts read: ll_dg3grid = LL@224 (shared grid), ll/nss/auc
    = 64 grid (ours native); nss_224/auc_224 added. One row per seed.
    The old runs/v2_eval_indomain logs + dg3compat logs scored a 1,000-image
    class-sorted prefix (2026-09-11) and are no longer read."""
    rows = []
    cb = _indomain_cb()
    for name, r, prov in _synth_fair_rows("ours"):
        m = re.match(r"ours_(\w+?)_seed(\d+)$", name)
        if not m or m.group(1) not in N_IMAGES:
            continue
        r64, r224 = r.get("64"), r.get("224")
        if not (r64 and r224):
            continue
        rows.append({"size": m.group(1), "n_images": N_IMAGES[m.group(1)],
                     "seed": int(m.group(2)), "selection": "best_val",
                     "ll_dg3grid": round(r224["ll_uniform_img"], 4),
                     "nss_224": round(r224["nss_img"], 4),
                     "auc_224": round(r224["auc_img"], 4),
                     "ll": round(r64["ll_uniform_img"], 4),
                     "nss": round(r64["nss_img"], 4),
                     "auc": round(r64["auc_img"], 4),
                     "n_fix": r224["n_fixations"], "n_img": r224["n_images"],
                     "amp": int(bool(prov.get("amp"))),
                     "ig_224": _ig(r224, cb, 224), "ig_64": _ig(r64, cb, 64)})
    rows.sort(key=lambda x: (x["n_images"], x["seed"]))
    write(RT / f"indomain_ours_v2{sfx()}.csv", rows)


def dg3_indomain():
    """In-domain DG3 rows from the same shared evaluator (native 224 map
    untouched at 224, area-pooled to 64). ll = LL@224 (what Fig 1 plots
    against ours' ll_dg3grid), nss/auc = 64 grid, nss_224/auc_224 added.
    nss_corrected=1 always (our NSS implementation, not DeepGaze's)."""
    rows = []
    cb = _indomain_cb()
    for name, r, prov in _synth_fair_rows("dg3"):
        # dg3_<size> = original run (rep 0); dg3_<size>_rep<k> = unseeded repetition (2026-09-20)
        m = re.match(r"dg3_(\w+?)(?:_rep(\d+))?$", name)
        if not m or m.group(1) not in N_IMAGES:
            continue
        r64, r224 = r.get("64"), r.get("224")
        if not (r64 and r224):
            continue
        rows.append({"size": m.group(1), "n_images": N_IMAGES[m.group(1)],
                     "rep": int(m.group(2)) if m.group(2) else 0,
                     "selection": "best_val",
                     "nss": round(r64["nss_img"], 4),
                     "ll": round(r224["ll_uniform_img"], 4),
                     "auc": round(r64["auc_img"], 4),
                     "nss_corrected": 1,
                     "ll_64": round(r64["ll_uniform_img"], 4),
                     "nss_224": round(r224["nss_img"], 4),
                     "auc_224": round(r224["auc_img"], 4),
                     "n_fix": r224["n_fixations"], "n_img": r224["n_images"],
                     "ig_224": _ig(r224, cb, 224), "ig_64": _ig(r64, cb, 64)})
    rows.sort(key=lambda x: (x["n_images"], x["rep"]))
    write(RT / f"indomain_dg3_v2{sfx()}.csv", rows)


def realism_subset():
    """Shared 400-image harness, one VALIDATION-selected checkpoint per model
    and rung (decided 2026-09-07):
      ScanDiff: every 5th-epoch checkpoint + last, scored at respacing 1000
                 on the validation subset; pick by mean rank of (MM, SM) KLD,
                 ties -> earlier (select_scandiff_ckpt). n_scored says how
                 much of the trajectory was available when the row was made.
      ours:     the best_val_ll checkpoint (the one rule for our model
                 everywhere); 'final' only as a labelled fallback until the
                 best_val_ll sampling has landed.
    The old mean-of-late-checkpoints estimator is retired (2026-09-07)."""
    src = SD / f"data/eval/v2_scoring_subset{sfx()}/rung_scores.csv"
    if not src.exists():
        print(f"SKIP realism: {src} missing")
        return
    ours = {}       # (size, train_seed) -> {ckpt: (mm, sm)}   sampling seed 0
    sd = {}         # (size, train_seed) -> {ckpt: (mm, sm)}   sampling seed 0
    reps = {}       # base label -> [(mm, sm) per sampling seed incl. 0]
    with src.open() as fh:
        for r in csv.DictReader(fh):
            label = r["label"]
            val = (float(r["mm_kld_nodur"]), float(r["sm_kld"]))
            base_label = re.sub(r"_s\d+$", "", label)
            reps.setdefault(base_label, []).append(val)
            if label != base_label:
                continue                      # select checkpoints using the original first-draw rule
            # ladder run is seed 42 (no suffix); further pretraining seeds carry
            # _seed<N> (sample_ours_subset.sh SEED=..., added 2026-09-17)
            m = re.match(r"ours_v2_(\w+?)_(final|best_val_ll)_T1(?:_seed(\d+))?$", label)
            if m and m.group(1) in N_IMAGES:
                tseed = int(m.group(3)) if m.group(3) else 42
                ours.setdefault((m.group(1), tseed), {})[m.group(2)] = val
            m = re.match(r"oursynth_v2_(\w+?)_seed(\d)_(epoch_\d+|last)_r1000$", label)
            if m and m.group(1) in N_IMAGES:
                sd.setdefault((m.group(1), int(m.group(2))), {})[m.group(3)] = val

    # Reporting only: average separately scored sampling draws within each run.
    # Checkpoint selection above/below remains based on the original validation scores.
    def band(base_label):
        v = reps.get(base_label, [])
        return {"mm_kld": sum(x[0] for x in v) / len(v),
                "sm_kld": sum(x[1] for x in v) / len(v),
                "sampling_aggregation": "mean_within_training_run",
                "n_sampling_seeds": len(v),
                "mm_kld_min": min(x[0] for x in v), "mm_kld_max": max(x[0] for x in v),
                "sm_kld_min": min(x[1] for x in v), "sm_kld_max": max(x[1] for x in v)}
    dg3 = {}
    with src.open() as fh:
        for r in csv.DictReader(fh):
            # dg3_v2_<size>_best_T1 = original run (train_seed 0);
            # dg3_v2_<size>_rep1_best_T1 = unseeded repetition (train_seed 1)
            m = re.match(r"dg3_v2_(\w+?)(?:_rep(\d+))?_best_T1$", r["label"])
            if m and m.group(1) in N_IMAGES:
                rep = int(m.group(2)) if m.group(2) else 0
                dg3[(m.group(1), rep)] = (float(r["mm_kld_nodur"]), float(r["sm_kld"]))
    if SPLIT == "test":
        # test rows exist only for the checkpoint chosen on validation; look
        # the choice up there rather than "selecting" among test rows
        vsrc = SD / "data/eval/v2_scoring_subset/rung_scores.csv"
        vsel = {}
        with vsrc.open() as fh:
            for r in csv.DictReader(fh):
                m = re.match(r"oursynth_v2_(\w+?)_seed(\d)_(epoch_\d+|last)_r1000$", r["label"])
                if m and m.group(1) in N_IMAGES:
                    vsel.setdefault((m.group(1), int(m.group(2))), {})[m.group(3)] = (
                        float(r["mm_kld_nodur"]), float(r["sm_kld"]))
        chosen = {k: select_scandiff_ckpt(d) for k, d in vsel.items()}
        sd = {k: {ck: d[ck]} for k, d in sd.items()
              if k in chosen and (ck := chosen[k]) in d}
    rows = []
    for (size, rep), d in dg3.items():
        lab = f"dg3_v2_{size}_rep{rep}_best_T1" if rep else f"dg3_v2_{size}_best_T1"
        rows.append({"model": "dg3", "size": size, "n_images": N_IMAGES[size], "train_seed": rep,
                     "ckpt": "best", "n_scored": 1, "mm_kld": d[0], "sm_kld": d[1],
                     **band(lab)})
    for (size, tseed), d in ours.items():
        ck = "best_val_ll" if "best_val_ll" in d else "final"
        lab = f"ours_v2_{size}_{ck}_T1" + (f"_seed{tseed}" if tseed != 42 else "")
        rows.append({"model": "ours", "size": size, "n_images": N_IMAGES[size], "train_seed": tseed,
                     "ckpt": ck, "n_scored": len(d),
                     "mm_kld": d[ck][0], "sm_kld": d[ck][1],
                     **band(lab)})
    for (size, tseed), d in sd.items():
        ck = select_scandiff_ckpt(d)
        rows.append({"model": "scandiff", "size": size, "n_images": N_IMAGES[size], "train_seed": tseed,
                     "ckpt": ck, "n_scored": len(d),
                     "mm_kld": d[ck][0], "sm_kld": d[ck][1],
                     **band(f"oursynth_v2_{size}_seed{tseed}_{ck}_r1000")})
    rows.sort(key=lambda r: (r["model"], r["n_images"], r["train_seed"]))
    write(RT / f"realism_subset_v2{sfx()}.csv", rows)


def peak_val(log: Path) -> dict | None:
    """Validation metrics at the best-LL epoch of a fine-tuning log.csv.

    ONE selection rule for our fine-tunes everywhere (decided 2026-09-03):
    best validation log-likelihood, i.e. the checkpoint train.py saves as
    best_val_ll.pt and the fair harness scores. NSS and AUC are read at that
    same epoch, never maximised separately (the earlier NSS-peak rule picked
    a different epoch in 23/36 runs, by <=0.026 NSS)."""
    if not log.exists():
        return None
    best = None
    with log.open() as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val":
                if best is None or float(r["ll"]) > best["ll"]:
                    best = {"nss": float(r["nss"]), "ll": float(r["ll"]),
                            "auc": float(r["auc"]),
                            "epoch": int(float(r["epoch"]))}
    return best


def transfer_ours():
    rows = []
    conds = ([("scratch", "scratch", 0)] +
             [(f"lad{s}", "av_scale", N_IMAGES[s]) for s in SIZES] +
             # init-rule robustness control: best_val_ll pretraining inits
             [(f"bv{s}", "av_scale_bv", N_IMAGES[s]) for s in SIZES] +
             [("sd5", "triangle", 100000), ("avfull", "triangle", 100000),
              ("avmatch", "triangle", 100000)])
    for cond, group, n in conds:
        for seed in (42, 43, 44, 45, 46):
            p = peak_val(REPO / f"runs/ft_authsplit_{cond}_seed{seed}/log.csv")
            if p:
                rows.append({"cond": cond, "group": group, "n_pretrain": n,
                             "seed": seed, "peak_nss": p["nss"],
                             "peak_ll": p["ll"], "peak_auc": p["auc"],
                             "peak_epoch": p["epoch"],
                             "final_nss": final_val(
                                 REPO / f"runs/ft_authsplit_{cond}_seed{seed}/log.csv")})
    write(RT / "transfer_ours_v2.csv", rows)


def final_val(log: Path) -> float | None:
    last = None
    with log.open() as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val":
                last = float(r["nss"])
    return last


def _scandiff_scores(base: Path, tag: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Duration-free scores only; never fall back to duration-inclusive keys.

    MultiMatch is the mean of four component KLDs, not the five-component
    MultiMatch aggregate. ScanMatchNoDur uses TempBin=0. Archived transfer
    scores use a fixed 512x384 frame with (height,width) metadata; upstream
    unpacking is correct here, unlike tools/score_scanpaths.py's W,H inputs.
    """
    runs: dict[str, dict[str, tuple[float, float]]] = {}
    for d in sorted(base.glob(f"*_{tag}")):
        m = re.match(rf"(.+)_(epoch_\d+|last)_{tag}$", d.name)
        if not m:
            continue
        f = next(d.glob("results/metrics_epoch_*/metrics_MIT1003Dataset_test.json"), None)
        if f is None:
            continue
        met = json.loads(f.read_text())
        components = [float(met[k]["KLD"]) for k in
                      ("MM_Vector", "MM_Direction", "MM_Length", "MM_Position")]
        sm = float(met["ScanMatchNoDur"]["KLD"])
        if not all(math.isfinite(v) for v in components + [sm]):
            raise ValueError(f"Non-finite duration-free metrics: {f}")
        runs.setdefault(m.group(1), {})[m.group(2)] = (sum(components) / 4, sm)
    return runs


def select_scandiff_ckpt(scores: dict[str, tuple[float, float]]) -> str:
    """ONE checkpoint per run, chosen on VALIDATION by the mean rank of
    (MM-KLD, SM-KLD), the same rule as the realism harness (decided
    2026-09-03; the earlier aggregator took two independent minima, i.e. a
    row that described no single model). Ties: earlier checkpoint."""
    ck = list(scores)
    def rank(idx):
        vals = sorted(scores[c][idx] for c in ck)
        return {c: vals.index(scores[c][idx]) for c in ck}
    r_mm, r_sm = rank(0), rank(1)
    order = {"last": 10**6}
    key = lambda c: ((r_mm[c] + r_sm[c]) / 2,
                     order.get(c, int(c.split("_")[1]) if c.startswith("epoch_") else 0))
    return min(ck, key=key)


def scandiff_ft():
    """One duration-free validation-selected checkpoint per ScanDiff run; both
    metrics from that checkpoint. In --split test mode the checkpoint is
    still chosen on validation and its test score is read; it is never
    selected on test."""
    val_runs = _scandiff_scores(SD / "data/eval/ft_valscore", "val")
    runs = {}
    for run, scores in val_runs.items():
        sel = select_scandiff_ckpt(scores)
        if SPLIT == "validation":
            mm, sm = scores[sel]
        else:
            test_scores = _scandiff_scores(SD / "data/eval/ft_testscore", "test").get(run, {})
            if sel not in test_scores:
                print(f"SKIP {run}: val-selected {sel} has no test score")
                continue
            mm, sm = test_scores[sel]
        runs[run] = {"best_mm": mm, "best_sm": sm, "ckpt": sel}
    rows = []
    labels = {
        "ft_mit_scratch": ("mit_only", "scratch", 0),
        "ft_mit_av1k": ("mit_only", "av_pretrained", 1000),
        "ft_mit_av10k": ("mit_only", "av_pretrained", 10000),
        "ft_mit_av50k": ("mit_only", "av_pretrained", 50000),
        "ft_mit_av100k": ("mit_only", "av_pretrained", 100000),
        "ft_mit_av200k": ("mit_only", "av_pretrained", 200000),
        "r1_official_recipe": ("joint", "scratch", 0),
        "armB_joint_avinit": ("joint", "av_pretrained", 100000),
    }
    for run, e in sorted(runs.items()):
        init = "selected" if run.endswith("_bvinit") else "last"
        run = run[:-len("_bvinit")] if init == "selected" else run
        m = re.match(r"(.+)_seed(\d+)$", run)
        if not m or m.group(1) not in labels:
            continue
        data, arm, n_pre = labels[m.group(1)]
        rows.append({"human_data": data, "arm": arm, "init": init if arm == "av_pretrained" else "-", "n_pretrain": n_pre,
                     "seed": int(m.group(2)), "ckpt": e["ckpt"],
                     "sm_kld": round(e["best_sm"], 4),
                     "mm_kld": round(e["best_mm"], 4)})
    write(RT / f"scandiff_ft_v2{sfx()}.csv", rows)


def ceiling():
    """Generator ceiling: the AV model that produced the corpus, replayed on
    the canonical val set (runs/fullval_eval/ceiling_fullval.log, all 16
    scanpaths/image). Same metric implementations as our eval; LL is our
    64-grid convention (no 224 lift exists for the generator)."""
    base = REPO / ("runs/fullval_eval_test" if SPLIT == "test" else "runs/fullval_eval")
    log = base / ("ceiling_fullval_224.log" if SPLIT == "test" else "ceiling_fullval.log")
    if not log.exists():
        return
    text = log.read_text()
    vals = {}
    for key, pat in (("ll", r"LL \(bits > unif\)\s*:\s*([0-9.]+)"),
                     ("nss", r"^\s*NSS\s*:\s*([0-9.]+)"),
                     ("auc", r"^\s*AUC\s*:\s*([0-9.]+)")):
        m = re.search(pat, text, re.M)
        if not m:
            return
        vals[key] = float(m.group(1))
    # 224-grid lift (slurm/nss_ceiling_224.sh): ll_dg3grid is on the same
    # grid as the ladder's ll_dg3grid / DG3's ll; nss_224/auc_224 are our
    # metrics on the lifted map (for the planned all-at-224 unification).
    lift = base / "ceiling_fullval_224.log"
    if lift.exists():
        t = lift.read_text()
        for key, pat in (("ll_dg3grid", r"LL@224 \(bits, DG3 grid\)\s*:\s*([0-9.]+)"),
                         ("nss_224", r"NSS@224\s*:\s*([0-9.]+)"),
                         ("auc_224", r"AUC@224\s*:\s*([0-9.]+)")):
            m = re.search(pat, t)
            if m:
                vals[key] = float(m.group(1))
    write(RT / f"indomain_ceiling{sfx()}.csv", [vals])


def parse_fair_log(path: Path) -> list[dict] | None:
    """Resolution table of eval_mit1003_fair.py / eval_mit1003_fair_dg3.py.

    Rows look like:
      64      2.194±0.000      2.175±0.000 ...
    Column order: res, LL_img, LL_fix, IG_cb_img, IG_cb_fix, CB_LL_img,
    NSS_img, NSS_fix, AUC_img, AUC_fix.
    """
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s+(64|128|224)\s+(.*)", line)
        if not m:
            continue
        vals = [float(v.split("±")[0]) for v in m.group(2).split()]
        if len(vals) != 9:
            continue
        keys = ["ll_img", "ll_fix", "ig_cb_img", "ig_cb_fix", "cb_ll_img",
                "nss_img", "nss_fix", "auc_img", "auc_fix"]
        rows.append({"res": int(m.group(1)), **dict(zip(keys, vals))})
    return rows or None


def fair_fixedsplit():
    """Fair-harness numbers for every evaluated ft_authsplit checkpoint plus
    the released-DG3 reference, all resolutions (224 is the primary one)."""
    conds = (["scratch"] + [f"lad{s}" for s in SIZES] + [f"bv{s}" for s in SIZES]
             + ["sd5", "avfull", "avmatch", "shuffled"])
    n_pre = {**{f"lad{s}": N_IMAGES[s] for s in SIZES},
             **{f"bv{s}": N_IMAGES[s] for s in SIZES},
             "scratch": 0, "sd5": 100000, "avfull": 100000,
             "avmatch": 100000, "shuffled": 100000}
    rows = []
    for cond in conds:
        for seed in (42, 43, 44, 45, 46):
            res_rows = parse_fair_log(
                REPO / f"runs/fair_fixedsplit{sfx()}/ours_{cond}_seed{seed}.log")
            for r in res_rows or []:
                rows.append({"model": "ours", "cond": cond,
                             "n_pretrain": n_pre[cond], "seed": seed, **r})
    # Frozen transfer runner outputs (validation AND test; 224 only,
    # image-averaged; provenance.json per record). Covers the bv ladder +
    # scratch (seeds 42/43), scratch/bv800k seeds 44-46 and the joint
    # MIT+COCO arms. In validation mode these duplicate the fair-log rows
    # (identical to 3 dp, checked 2026-09-11) and are only added for conds
    # the fair logs do not have; in test mode they are the only ours source.
    def _frozen_rows(base: Path, model: str):
        out = []
        d = base / ("test" if SPLIT == "test" else "validation")
        for f in sorted(d.glob("*/metrics.json")):
            name = f.parent.name                       # ours_bv800k_seed44
            m = re.match(r"ours_(joint_)?(\w+?)_seed(\d+)$", name)
            if not m:
                continue
            cond = ("joint_" if m.group(1) else "") + m.group(2)
            seed = int(m.group(3))
            n = n_pre.get(m.group(2), 0)
            for res, r in json.loads(f.read_text())["results"].items():
                out.append({"model": model, "cond": cond, "n_pretrain": n,
                            "seed": seed, "res": int(res),
                            "ll_img": round(r["ll_uniform_img"], 4),
                            "ll_fix": round(r["ll_uniform_fix"], 4),
                            "ig_cb_img": round(r["ig_centerbias_img"], 4),
                            "ig_cb_fix": round(r["ig_centerbias_fix"], 4),
                            "cb_ll_img": round(r["centerbias_ll_uniform_img"], 4),
                            "nss_img": round(r["nss_img"], 4),
                            "nss_fix": round(r["nss_fix"], 4),
                            "auc_img": round(r["auc_img"], 4),
                            "auc_fix": round(r["auc_fix"], 4)})
        return out
    have = {(r["cond"], r["seed"], r["res"]) for r in rows}
    for base in (REPO / "runs/frozen_transfer_ours_20260908",
                 REPO / "runs/frozen_transfer_extra_20260910",
                 REPO / "runs/frozen_triangle_20260912"):
        for r in _frozen_rows(base, "ours"):
            if (r["cond"], r["seed"], r["res"]) not in have:
                rows.append(r); have.add((r["cond"], r["seed"], r["res"]))
    for r in parse_fair_log(
            SCANPATHER / f"runs/fair_fixedsplit{sfx()}/dg3_released_fair.log") or []:
        rows.append({"model": "dg3_released", "cond": "released",
                     "n_pretrain": -1, "seed": -1, **r})
    # DG3 fine-tuned inside the protocol (score_dg3_fixedsplit_fair.sh)
    # 'Nk' = spatial-only synthetic transfer (secondary), 'fullNk' =
    # scanpath-module transfer (primary since 2026-09-03)
    dg3_pre = {"scratch": 0, "salicon": -2, "1k": 1000, "10k": 10000,
               "50k": 50000, "100k": 100000, "200k": 200000,
               "full1k": 1000, "full10k": 10000, "full50k": 50000, "full100k": 100000, "full200k": 200000}
    # DG3 fine-tunes: STAGE-3 checkpoints from the frozen transfer runner
    # (runs/frozen_transfer_dg3_20260909/{validation,test}/<id>/metrics.json,
    # 224 grid, image-averaged; provenance.json carries checkpoint hash and
    # stage). The scanpather fair logs (dg3_fs_*.log) scored STAGE 2 because
    # score_dg3_fixedsplit_fair.sh never passed --stage and the evaluator
    # defaults to the partially-frozen stage (found 2026-09-11).
    # They are no longer read here.
    frozen = REPO / "runs/frozen_transfer_dg3_20260909" / (
        "test" if SPLIT == "test" else "validation")
    for cond, n in dg3_pre.items():
        if not cond.startswith(("scratch", "salicon", "full")):
            continue  # spatial-only transfer arms were never frozen-scored
        for rep in (0, 1):
            f = frozen / f"dg3_{cond}_rep{rep}" / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())["results"]
            for res, r in m.items():
                rows.append({"model": "dg3_ft", "cond": cond, "n_pretrain": n,
                             "seed": rep, "res": int(res),
                             "ll_img": round(r["ll_uniform_img"], 4),
                             "ll_fix": round(r["ll_uniform_fix"], 4),
                             "ig_cb_img": round(r["ig_centerbias_img"], 4),
                             "ig_cb_fix": round(r["ig_centerbias_fix"], 4),
                             "cb_ll_img": round(r["centerbias_ll_uniform_img"], 4),
                             "nss_img": round(r["nss_img"], 4),
                             "nss_fix": round(r["nss_fix"], 4),
                             "auc_img": round(r["auc_img"], 4),
                             "auc_fix": round(r["auc_fix"], 4)})
    write(RT / f"fair_fixedsplit_v2{sfx()}.csv", rows)


def zeroshot():
    """Zero-shot ladder (pretraining checkpoints, no fine-tuning) through the
    fair harness (runs/fair_fixedsplit/zeroshot_<size>.log)."""
    rows = []
    for s in SIZES:
        for r in parse_fair_log(
                REPO / f"runs/fair_fixedsplit{sfx()}/zeroshot_{s}.log") or []:
            rows.append({"size": s, "n_images": N_IMAGES[s], **r})
    write(RT / f"zeroshot_ours_v2{sfx()}.csv", rows)


def xfam_mit():
    """Cross-family whole-scanpath comparison on MIT1003 (2026-09-14): every
    model's fine-tuned checkpoints sampled/generated on the split's images
    and scored by tools/score_scanpaths.py against the human reference WITH
    initial fixation (mit1003_human_initial_<split>.json). ScanMatch = the
    width/height-corrected variant (MIT images are non-square)."""
    src = SD / f"data/eval/mit_xfam_{SPLIT}/scores.csv"
    if not src.exists():
        print(f"SKIP xfam: {src} missing")
        return
    rows = []
    sources = [src]
    # The ScanDiff joint-training follow-up replaces the earlier 100k
    # initialization with the best MIT1003-only scale (1k).  It was scored
    # into a frozen, standalone CSV so the original aggregate stayed intact.
    joint_1k = (SD / "paper_reproduction/joint_av1k_followup_20260923"
                / "xfam_test_scores.csv")
    if SPLIT == "test" and joint_1k.exists():
        sources.append(joint_1k)
    reps = {}
    # validation-selected checkpoint per ScanDiff fine-tune run (2026-09-20): scores.csv
    # accumulates every checkpoint ever rescored; only the selected one may enter the tables.
    sd_selected = {run: select_scandiff_ckpt(sc)
                   for run, sc in _scandiff_scores(SD / "data/eval/ft_valscore", "val").items()}
    for source in sources:
        with source.open() as fh:
            for r in csv.DictReader(fh):
                b = re.sub(r"_s\d+$", "", r["label"])
                reps.setdefault(b, []).append(tuple(float(r[k]) for k in
                    ("mm_kld_nodur", "sm_kld_corr", "sm_kld", "mm_sim_model", "mm_sim_human")))
    for source in sources:
        with source.open() as fh:
          for r in csv.DictReader(fh):
            lab = r["label"]
            if re.search(r"_s\d+$", lab):
                continue
            if (m := re.match(r"ours_ft_(\w+?)_seed(\d+)_T1$", lab)):
                cond, seed = m.group(1), int(m.group(2))
                # joint_<cond>: MIT1003 + COCO-FreeView fine-tunes
                # (slurm/sample_ours_mit_ft_joint.sh, 2026-09-16)
                # jointv2_<cond> = joint runs with the COCO start-fixation fix (2026-09-19);
                # the older joint_<cond> rows are superseded and skipped.
                if cond.startswith("joint_"):
                    continue
                hd = "joint" if cond.startswith("jointv2_") else "mit_only"
                cond = cond[8:] if hd == "joint" else cond
                n = 0 if cond == "scratch" else N_IMAGES[cond[2:]]
                row = {"model": "ours", "human_data": hd, "cond": cond, "n_pretrain": n, "seed": seed, "ckpt": "best_val_ll", "init": "selected" if cond != "scratch" else "-"}
            elif (m := re.match(r"dg3_ft_(\w+?)_rep(\d)_T1$", lab)):
                cond, seed = m.group(1), int(m.group(2))
                n = {"scratch": 0, "salicon": -2}.get(cond, N_IMAGES.get(cond[4:], 0))
                row = {"model": "dg3", "human_data": "mit_only", "cond": cond, "n_pretrain": n, "seed": seed, "ckpt": "stage3_best", "init": "-"}
            elif lab == "scandiff_released_freeview":
                row = {"model": "scandiff", "human_data": "joint", "cond": "released", "n_pretrain": -1, "seed": -1, "ckpt": "released", "init": "-"}
            elif (m := re.match(r"scandiff_ft_(\w+?)_(epoch_\d+|last)(_lenmatch)?$", lab)):
                run, ck, lenmatch = m.group(1), m.group(2), bool(m.group(3))
                if run.startswith("ft_mit_scratch"): hd, cond, n = "mit_only", "scratch", 0
                elif run.startswith("ft_mit_av"): hd, cond = "mit_only", "av_pretrained"; n = N_IMAGES[re.search(r"av(\w+?)_seed", run).group(1)]
                elif run.startswith("r1_official"): hd, cond, n = "joint", "scratch", 0
                elif run.startswith("armB_joint"):
                    # When the completed 1k follow-up is available, discard
                    # the superseded 100k joint-training condition.
                    if joint_1k in sources and "_av1k_" not in run:
                        continue
                    hd, cond = "joint", "av_pretrained"
                    n = 1000 if "_av1k_" in run else 100000
                else: continue
                seed = int(re.search(r"seed(\d)", run).group(1))
                init = "selected" if run.endswith("_bvinit") else "last"
                if cond == "av_pretrained" and init == "last":
                    continue                     # discarded arm (last-epoch init, superseded 2026-09-19)
                if not lenmatch and sd_selected.get(run) != ck:
                    continue                     # not the validation-selected checkpoint
                row = {"model": "scandiff", "human_data": hd, "cond": cond + ("_lenmatch" if lenmatch else ""),
                       "n_pretrain": n, "seed": seed, "ckpt": ck, "init": init if cond == "av_pretrained" else "-"}
            else:
                continue
            v = reps[lab]
            row.update({"mm_kld": sum(x[0] for x in v) / len(v),
                        "sm_kld_corr": sum(x[1] for x in v) / len(v),
                        "sampling_aggregation": "mean_within_training_run",
                        "n_sampling_seeds": len(v),
                        "mm_kld_min": round(min(x[0] for x in v), 5), "mm_kld_max": round(max(x[0] for x in v), 5),
                        "sm_kld_corr_min": round(min(x[1] for x in v), 5), "sm_kld_corr_max": round(max(x[1] for x in v), 5),
                        "sm_kld_upstream": sum(x[2] for x in v) / len(v),
                        "mm_sim": sum(x[3] for x in v) / len(v),
                        "mm_sim_ref": sum(x[4] for x in v) / len(v)})
            rows.append(row)
    rows.sort(key=lambda x: (x["model"], x["human_data"], x["n_pretrain"], x["cond"], x["seed"]))
    write(RT / f"xfam_mit_v2{sfx()}.csv", rows)


def write(path: Path, rows: list[dict]):
    if not rows:
        print(f"SKIP {path.name}: no rows")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="'test' = final evaluation only: read the *_test evaluation "
                         "outputs and write *_test.csv (validation tables untouched)")
    SPLIT = ap.parse_args().split
    RT.mkdir(parents=True, exist_ok=True)
    ours_indomain()
    dg3_indomain()
    realism_subset()
    scandiff_ft()
    fair_fixedsplit()
    zeroshot()
    xfam_mit()
    ceiling()
    if SPLIT == "validation":
        # selection-side table: training-log peaks (authors' val split)
        transfer_ours()
