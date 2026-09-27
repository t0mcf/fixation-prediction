"""Post-hoc validation loss for saved ScanDiff checkpoints.

The trainer never invoked its validation() method during training (no
evaluator was configured, since test=false), so no validation signal exists
for the 1k/10k(_v2)/50k/100k/200k runs. This script replays that same
diffusion training-loss computation (one random timestep per sample, same
loss_dict as train/loss) on the canonical validation split for every saved
checkpoint, so we can see whether training loss and val loss diverge
(overfitting) at any scale.

Uses a much larger validation batch size than the config default (1, which
exists only for the sampling-based evaluator) since this is a plain forward
loss, not scanpath generation.

Usage:
  python scripts/eval_val_loss_checkpoints.py --size 1k
  python scripts/eval_val_loss_checkpoints.py --size 10k_v2 --batch-size 128
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ.setdefault("HF_HOME", "/mnt/vast-nhr/projects/nim00018/tom/.cache/huggingface")

import rootutils
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.create_diffusion import create_diffusion
from src.utils.instantiators import instantiate_datasets, instantiate_collators
from src.data.unified_datamodule import UnifiedDataModule
import hydra


CONFIG_DIR = str(ROOT / "configs")


def build_cfg(size: str, batch_size_validation: int):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"data/val_datasets=[oursynth_{size}]",
                f"data/train_datasets=[oursynth_{size}]",
                f"data/test_datasets=[oursynth_{size}]",
                f"data.batch_size_validation={batch_size_validation}",
                "trainer=gpu",
                "callbacks=default",
                "train=false",
                "test=false",
                "diffusion.num_timesteps=1000",
                "diffusion_class=spaced_diffusion",
                f"hydra.run.dir=./runs_ours/eval_val_loss_{size}",
            ],
        )
    return cfg


@torch.no_grad()
def eval_checkpoint(model, diffusion, val_loader, device, max_batches=None):
    model.eval()
    all_losses = []
    for step, batch in enumerate(tqdm(val_loader, desc="val", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        x = batch["scanpath"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        task_embedding = batch.get("task_embedding")
        if task_embedding is not None:
            task_embedding = task_embedding.to(device)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(
            y=batch["img"].to(device),
            padding_mask=padding_mask,
            task_embedding=task_embedding,
        )
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        all_losses.append(loss.item())
    model.train()
    return float(np.mean(all_losses)), len(all_losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, help="e.g. 1k, 10k_v2, 50k, 100k, 200k")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None,
                         help="cap val batches per checkpoint for a quick pass")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt_dir = ROOT / f"runs_ours/train_{args.size}/train/checkpoints"
    ckpts = sorted(
        glob.glob(str(ckpt_dir / "epoch_*.pt")),
        key=lambda p: int(re.search(r"epoch_(\d+)\.pt", p).group(1)),
    )
    assert ckpts, f"no checkpoints found in {ckpt_dir}"
    print(f"Found {len(ckpts)} checkpoints for oursynth_{args.size}")

    cfg = build_cfg(args.size, args.batch_size)

    datasets = instantiate_datasets(cfg.get("data"))
    collators = instantiate_collators(cfg.get("data"))
    to_dict_config = OmegaConf.to_container(cfg.data)
    for k in ["train_datasets", "val_datasets", "test_datasets",
              "train_collators", "val_collators", "test_collators"]:
        to_dict_config.pop(k, None)
    datamodule = UnifiedDataModule(
        train_datasets=datasets["train_datasets"],
        val_datasets=datasets["val_datasets"],
        test_datasets=datasets["test_datasets"],
        train_collators=collators["train_collators"],
        val_collators=collators["val_collators"],
        test_collators=collators["test_collators"],
        **to_dict_config,
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    print(f"Validation set: {len(datamodule.data_val)} scanpaths, "
          f"{len(val_loader)} batches at batch_size={args.batch_size}")

    model = hydra.utils.instantiate(cfg.model).to(args.device)
    diffusion = create_diffusion(
        cfg, timestep_respacing="",
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )

    results = []
    for ckpt_path in ckpts:
        epoch = int(re.search(r"epoch_(\d+)\.pt", ckpt_path).group(1))
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        mean_loss, n_batches = eval_checkpoint(
            model, diffusion, val_loader, args.device, max_batches=args.max_batches
        )
        print(f"epoch {epoch:4d}  val/mean_loss = {mean_loss:.4f}  ({n_batches} batches)")
        results.append({"epoch": epoch, "val_loss": mean_loss, "n_batches": n_batches})

    out_path = args.out or str(ROOT / f"runs_ours/train_{args.size}/val_loss_curve.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
