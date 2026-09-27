"""Generate scanpaths from a v2 ScanDiff checkpoint on the FROZEN scoring
subset (oursynth_v2_sub400, built by tools/make_v2_scoring_subset.py).

One checkpoint -> one generations pkl (evaluator format, reusable by
tools/scandiff_gen_to_npy.py) plus the spatial-mean line, so every scoring
pass doubles as a displacement check. 400 images x 5 paths; at --respacing
250 roughly 30 min on an A100, at 1000 (final numbers) ~2 h.

  python scripts/generate_v2_subset.py \
      --run-dir runs_thesis_v2/oursynth_v2_50k_seed0 --ckpt epoch_60.pt \
      --respacing 250 --out <dir>
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.components.collator import Collator
from src.data.unified_datamodule import UnifiedDataModule
from src.evaluator import Evaluator
from src.utils.create_diffusion import create_diffusion

ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
CONFIG_DIR = str(ROOT / "configs")
NAME = "oursynth_v2_sub400"
PAD = -3.0


def main():
    global NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--respacing", type=int, default=250)
    ap.add_argument("--out", required=True, help="output directory for the pkl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bn-mode", choices=["eval", "recalib"], default="eval",
                    help="eval = sample with the checkpoint's BatchNorm running "
                         "statistics (upstream behaviour). recalib = DIAGNOSTIC "
                         "(2026-09-04): re-estimate every BatchNorm running "
                         "mean/var on --recalib-batches training batches of the "
                         "run's own dataset (train mode, no_grad, cumulative "
                         "average, no weight update) before sampling in eval "
                         "mode. Tests the train/sample BN-mismatch hypothesis.")
    ap.add_argument("--recalib-batches", type=int, default=200)
    ap.add_argument("--limit-frac", type=float, default=1.0,
                    help="fraction of the subset to sample (quick CPU diagnostics)")
    ap.add_argument("--recalib-dataset", default=None,
                    help="hydra dataset name of the run's training data "
                         "(default: derived from --run-dir, e.g. oursynth_v2_100k)")
    ap.add_argument("--gen-seed", type=int, default=None,
                    help="torch.manual_seed before sampling; default None = "
                         "unseeded (the historical behaviour: every run draws "
                         "fresh initial noise). Set for labelled replicates.")
    ap.add_argument("--subset-name", default=NAME,
                    help="hydra dataset name of the scoring subset "
                         "(oursynth_v2_sub400 = validation side; "
                         "oursynth_v2_sub400_test = test side, final evaluation only)")
    args = ap.parse_args()
    NAME = args.subset_name

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"data/train_datasets=[{NAME}]", f"data/val_datasets=[{NAME}]",
                f"data/test_datasets=[{NAME}]",
                "trainer=gpu", "callbacks=default", "train=false", "test=false",
                "diffusion.num_timesteps=1000", "diffusion_class=spaced_diffusion",
                f"hydra.run.dir={out_dir}/hydra",
            ],
        )

    collator = Collator(max_len=16, use_abs_coords=True, time_in_ms=False)
    data_cfg = OmegaConf.to_container(cfg.data)
    datasets = {k: data_cfg.pop(k) for k in
                ["train_datasets", "val_datasets", "test_datasets"]}
    for k in ["train_collators", "val_collators", "test_collators"]:
        data_cfg.pop(k, None)
    data_cfg["batch_size_validation"] = 1
    val_dataset = hydra.utils.instantiate(datasets["val_datasets"][NAME])
    print(f"subset rows: {len(val_dataset)}", flush=True)

    datamodule = UnifiedDataModule(
        train_datasets=[val_dataset], val_datasets=[val_dataset],
        test_datasets=[val_dataset],
        train_collators=[collator], val_collators=[collator],
        test_collators=[collator], **data_cfg,
    )
    datamodule.setup()

    model = hydra.utils.instantiate(cfg.model).to(args.device)
    ckpt_path = run_dir / "train/checkpoints" / args.ckpt
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.bn_mode == "recalib":
        import re as _re
        ds_name = args.recalib_dataset or _re.sub(r"_seed\d+.*$", "", run_dir.name)
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg_tr = compose(config_name="train.yaml", overrides=[
                f"data/train_datasets=[{ds_name}]", f"data/val_datasets=[{ds_name}]",
                f"data/test_datasets=[{ds_name}]", "trainer=gpu", "callbacks=default",
                "train=false", "test=false", "diffusion.num_timesteps=1000",
                "diffusion_class=spaced_diffusion", f"hydra.run.dir={out_dir}/hydra_recalib"])
        tr_cfg = OmegaConf.to_container(cfg_tr.data)
        tr_sets = {k: tr_cfg.pop(k) for k in ["train_datasets", "val_datasets", "test_datasets"]}
        for k in ["train_collators", "val_collators", "test_collators"]:
            tr_cfg.pop(k, None)
        train_ds = hydra.utils.instantiate(tr_sets["train_datasets"][ds_name])
        tr_dm = UnifiedDataModule(train_datasets=[train_ds], val_datasets=[train_ds],
                                  test_datasets=[train_ds], train_collators=[collator],
                                  val_collators=[collator], test_collators=[collator], **tr_cfg)
        tr_dm.setup()
        diffusion_train = create_diffusion(
            cfg, timestep_respacing="", diffusion_steps=cfg.diffusion.num_timesteps,
            noise_schedule=cfg.diffusion.noise_schedule, predict_xstart=cfg.diffusion.predict_xstart)
        bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        before = {n: (float(m.running_var.mean()), int(m.num_batches_tracked))
                  for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        for m in bns:
            m.reset_running_stats(); m.momentum = None      # cumulative average over the pass
        model.train()
        with torch.no_grad():
            for i, batch in enumerate(tr_dm.train_dataloader()):
                if i >= args.recalib_batches:
                    break
                x = batch["scanpath"].to(args.device)
                mk = dict(y=batch["img"].to(args.device),
                          padding_mask=batch["padding_mask"].to(args.device),
                          task_embedding=(batch["task_embedding"].to(args.device)
                                          if "task_embedding" in batch else None))
                t = torch.randint(0, diffusion_train.num_timesteps, (x.shape[0],), device=args.device)
                diffusion_train.training_losses(model, x, t, mk)   # forward only: updates BN stats
        for m in bns:
            m.momentum = 0.1
        model.eval()
        after = {n: (float(m.running_var.mean()), int(m.num_batches_tracked))
                 for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        print(f"BN recalibration on {args.recalib_batches} batches of {ds_name}:", flush=True)
        for n in before:
            print(f"  {n:32s} running_var {before[n][0]:.4f} -> {after[n][0]:.4f}  "
                  f"(nbt {before[n][1]} -> {after[n][1]})", flush=True)
    epoch = ckpt.get("global_epoch")
    print(f"checkpoint {ckpt_path} (epoch {epoch})", flush=True)

    diffusion = create_diffusion(
        cfg, timestep_respacing=str(args.respacing),
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )
    evaluator = hydra.utils.instantiate(
        cfg.evaluation, datamodule=datamodule,
        eval_root_path=str(out_dir), _target_=Evaluator)
    evaluator.limit_test_batches = args.limit_frac

    if args.gen_seed is not None:
        torch.manual_seed(args.gen_seed)
        print(f"generation seed: {args.gen_seed}", flush=True)
    preds, _ = evaluator.extract_predictions(model, diffusion, epoch=0,
                                             is_validation=True)

    # detach to CPU numpy so the pkl is loadable anywhere
    clean = {}
    points = []
    for stem, record in preds.items():
        sps = record["scanpaths"]
        sps = (sps.detach().cpu().numpy()
               if torch.is_tensor(sps) else np.asarray(sps))
        clean[stem] = {"scanpaths": sps}
        for sp in sps:
            keep = ~np.any(sp[:, :2] <= PAD + 1e-3, axis=1)
            if keep.sum() >= 2:
                points.append(sp[keep, :2])
    pkl = out_dir / f"generations_{run_dir.name}_{args.ckpt.replace('.pt','')}.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(clean, fh)

    p = np.concatenate(points)
    print(f"wrote {pkl}")
    print(f"spatial mean position: ({p[:,0].mean():.3f}, {p[:,1].mean():.3f})  "
          f"sd ({p[:,0].std():.3f}, {p[:,1].std():.3f})  "
          f"[reference: centered ~ (0.5, 0.5); failed retrain (0.352, 0.364)]")


if __name__ == "__main__":
    main()
