"""Spatial sanity check for the protocol-v2 ScanDiff ladder checkpoints.

The MIT1003 retrain of 2026-08-20 failed in a specific way: its generated
scanpaths kept a normal length and spread but sat about 15% of the image up and
to the left of where they belong -- mean position (0.352, 0.364) instead of the
released checkpoint's (0.478, 0.523). The thesis-v2 ladder is trained with the
same settings as that run, so it has to be checked for the same symptom.

The ladder trains with test=false, which disables the evaluator, so no sampled
scanpaths exist. This samples a small number of images per checkpoint and
reports where the fixations land. It answers one question only: is a ladder
checkpoint spatially displaced? It is not a scoring run.

  python scripts/check_v2_spatial.py --size 10k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The repo is not installed as a package and nothing on the job's path points at
# it, so `import src...` fails when the script runs from SLURM.
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
PAD = -3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="1k, 10k, 50k, 100k or 200k")
    ap.add_argument("--respacing", type=int, default=1000)
    ap.add_argument("--limit", type=float, default=0.02,
                    help="fraction of validation batches to sample")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-dir", default=None,
                    help="run directory containing train/checkpoints/; "
                         "default runs_thesis_v2/oursynth_v2_{size}_seed0")
    ap.add_argument("--ckpt", default="last.pt",
                    help="checkpoint filename inside train/checkpoints/")
    args = ap.parse_args()

    name = f"oursynth_v2_{args.size}"
    run_dir = Path(args.run_dir) if args.run_dir else (
        ROOT / f"runs_thesis_v2/{name}_seed0")
    out_dir = ROOT / f"runs_ours/spatialcheck_{run_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"data/train_datasets=[{name}]", f"data/val_datasets=[{name}]",
                f"data/test_datasets=[{name}]",
                "trainer=gpu", "callbacks=default", "train=false", "test=false",
                "diffusion.num_timesteps=1000", "diffusion_class=spaced_diffusion",
                f"hydra.run.dir={out_dir}",
            ],
        )

    collator = Collator(max_len=16, use_abs_coords=True, time_in_ms=False)
    data_cfg = OmegaConf.to_container(cfg.data)
    datasets = {k: data_cfg.pop(k) for k in
                ["train_datasets", "val_datasets", "test_datasets"]}
    for k in ["train_collators", "val_collators", "test_collators"]:
        data_cfg.pop(k, None)
    data_cfg["batch_size_validation"] = 1
    val_dataset = hydra.utils.instantiate(datasets["val_datasets"][name])
    print(f"validation rows: {len(val_dataset)}", flush=True)

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
    print(f"checkpoint {ckpt_path.name}, epoch {ckpt.get('global_epoch')}", flush=True)

    diffusion = create_diffusion(
        cfg, timestep_respacing=str(args.respacing),
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )
    evaluator = hydra.utils.instantiate(
        cfg.evaluation, datamodule=datamodule,
        eval_root_path=str(out_dir), _target_=Evaluator)
    evaluator.limit_test_batches = args.limit

    preds, _ = evaluator.extract_predictions(model, diffusion, epoch=0, is_validation=True)

    points, lengths = [], []
    for record in preds.values():
        # extract_predictions hands back CUDA tensors; np.asarray on one raises.
        scanpaths = record["scanpaths"]
        scanpaths = (scanpaths.detach().cpu().numpy()
                     if torch.is_tensor(scanpaths) else np.asarray(scanpaths))
        for sp in scanpaths:
            keep = ~np.any(sp[:, :2] <= PAD + 1e-3, axis=1)
            xy = sp[keep, :2]
            if len(xy) >= 2:
                points.append(xy)
                lengths.append(len(xy))
    p = np.concatenate(points)
    print(f"\n=== v2 {args.size}, epoch {ckpt.get('global_epoch')} ===")
    print(f"  images sampled   : {len(preds)}   scanpaths: {len(lengths)}")
    print(f"  mean position    : ({p[:, 0].mean():.3f}, {p[:, 1].mean():.3f})")
    print(f"  sd               : ({p[:, 0].std():.3f}, {p[:, 1].std():.3f})")
    print(f"  mean length      : {np.mean(lengths):.2f}")
    print("  reference -- released ScanDiff on MIT1003: (0.478, 0.523)")
    print("  reference -- failed 2026-08-20 retrain    : (0.352, 0.364)")


if __name__ == "__main__":
    main()
