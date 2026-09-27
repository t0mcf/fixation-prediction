"""
Phase 2 of ScanDiff-as-data-generator: generate scanpaths with the
real-MIT1003-trained teacher (train_mit1003_t90) on the oursynth_10k ImageNet
images, and write them as a parquet in fixation-prediction's training schema:

  image_path : './data/ImageNet/train_images/<class>/<stem>.JPEG'
  epoch      : int16 scanpath index (0..sp_per_image-1)
  locations  : flat list<float> of (x, y) pairs in [-1, 1], first pair (0, 0)
               (center start, matching the active-vision data convention);
               VARIABLE length — the teacher terminates naturally, and
               fixation-prediction's dataset.py pads/truncates itself.
  log_pis    : empty list (unused by dataset.py; kept for schema parity)

Sharding: images are split deterministically; run as an array job:
  python scripts/generate_pretrain_scandiff.py --dataset oursynth_10k \
      --shard $SLURM_ARRAY_TASK_ID --num-shards 8 \
      --out data/scandiffgen_10k/shard_$SLURM_ARRAY_TASK_ID.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra

import rootutils
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.create_diffusion import create_diffusion
from src.data.components.mit1003.mit1003 import MIT1003Dataset
from src.data.components.collator import Collator
from src.data.unified_datamodule import UnifiedDataModule
from src.evaluator import Evaluator

CONFIG_DIR = str(ROOT / "configs")
PAD = -3.0


def build_cfg(dataset: str, run_tag: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(
            config_name="train.yaml",
            overrides=[
                f"data/val_datasets=[{dataset}]",
                f"data/train_datasets=[{dataset}]",
                f"data/test_datasets=[{dataset}]",
                "trainer=gpu", "callbacks=default",
                "train=false", "test=false",
                "diffusion.num_timesteps=1000", "diffusion_class=spaced_diffusion",
                f"hydra.run.dir=./runs_ours/{run_tag}",
            ],
        )


def stem_to_imagenet_path(stem: str) -> str:
    cls = stem.split("_")[0]
    return f"./data/ImageNet/train_images/{cls}/{stem}.JPEG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oursynth_10k",
                    help="scandiff dataset dir name (features must exist)")
    ap.add_argument("--teacher-ckpt",
                    default=str(ROOT / "runs_ours/train_mit1003_t90/train/checkpoints/last.pt"))
    ap.add_argument("--sp-per-image", type=int, default=16)
    ap.add_argument("--respacing", type=int, default=1000,
                    help="diffusion sampling steps (1000 = full fidelity)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds_dir = ROOT / "data" / args.dataset
    train_json = ds_dir / f"{args.dataset}_fixations_train.json"
    rows = json.load(open(train_json))
    stems = sorted({Path(r["name"]).stem for r in rows})
    shard_stems = stems[args.shard::args.num_shards]
    print(f"{len(stems)} images total; shard {args.shard}/{args.num_shards} -> {len(shard_stems)} images")

    # generation spec: sp_per_image dummy rows per image in this shard.
    # X/Y/T are placeholders (generation conditions on image features + task
    # only); width/height copied from any real row of that image.
    meta = {}
    for r in rows:
        meta.setdefault(Path(r["name"]).stem, r)
    spec = []
    for stem in shard_stems:
        m = meta[stem]
        for j in range(args.sp_per_image):
            spec.append({
                "name": m["name"], "split": "test",
                "width": m["width"], "height": m["height"],
                "X": [m["width"] / 2.0], "Y": [m["height"] / 2.0], "T": [200.0],
                "subject": j,
            })
    run_tag = f"gen_{args.dataset}_shard{args.shard}"
    spec_dir = ROOT / "runs_ours" / run_tag / "specdata"
    spec_dir.mkdir(parents=True, exist_ok=True)
    specname = f"genspec_{args.shard}"
    json.dump(spec, open(spec_dir / f"{specname}_fixations_test.json", "w"))
    feat = spec_dir / "dinov2_base_timm_image_features"
    if not feat.exists():
        feat.symlink_to((ds_dir / "dinov2_base_timm_image_features").resolve())

    cfg = build_cfg(args.dataset, run_tag)
    gen_dataset = MIT1003Dataset(
        name=specname, root_path=str(spec_dir), task="What do you see in the image?",
        split="test", num_subjects=args.sp_per_image, time_in_ms=False,
        use_abs_coords=True, task_embeddings_file="task_embeddings.npy",
        img_features_dir="dinov2_base_timm_image_features",
    )
    print(f"generation spec rows: {len(gen_dataset)}")

    collator = Collator(max_len=16, use_abs_coords=True, time_in_ms=False)
    dm_cfg = OmegaConf.to_container(cfg.data)
    for k in ["train_datasets", "val_datasets", "test_datasets",
              "train_collators", "val_collators", "test_collators"]:
        dm_cfg.pop(k, None)
    datamodule = UnifiedDataModule(
        train_datasets=[gen_dataset], val_datasets=[gen_dataset], test_datasets=[gen_dataset],
        train_collators=[collator], val_collators=[collator], test_collators=[collator],
        **dm_cfg,
    )
    datamodule.setup()

    model = hydra.utils.instantiate(cfg.model).to(args.device)
    ckpt = torch.load(args.teacher_ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"teacher loaded: {args.teacher_ckpt} (epoch {ckpt.get('global_epoch')})")

    diffusion = create_diffusion(
        cfg, timestep_respacing=str(args.respacing),
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )
    evaluator = hydra.utils.instantiate(
        cfg.evaluation, datamodule=datamodule,
        eval_root_path=str(ROOT / "runs_ours" / run_tag), _target_=Evaluator,
    )
    evaluator.limit_test_batches = 1.0

    preds, _ = evaluator.extract_predictions(model, diffusion, epoch=0, is_validation=True)
    print(f"generated for {len(preds)} images")

    out_rows = []
    n_sp = 0
    for name, rec in preds.items():
        stem = Path(name).stem
        sps = rec["scanpaths"]
        sps = sps.cpu().numpy() if torch.is_tensor(sps) else np.asarray(sps)
        for j, sp in enumerate(sps):
            valid = ~np.any(sp[:, :2] <= PAD + 1e-3, axis=1)
            xy = sp[valid, :2]
            if len(xy) < 2:
                continue
            xy = xy * 2.0 - 1.0                       # [0,1] -> [-1,1]
            # center start + up to 15 teacher fixations = same budget as the
            # active-vision rows (16 pairs max)
            loc = np.concatenate([np.zeros((1, 2), np.float32),
                                  xy[:15].astype(np.float32)])
            out_rows.append({
                "image_path": stem_to_imagenet_path(stem),
                "epoch": np.int16(j),
                "locations": loc.reshape(-1).tolist(),
                "log_pis": [],
            })
            n_sp += 1

    df = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    lens = df["locations"].map(len)
    print(f"wrote {n_sp} scanpaths / {df['image_path'].nunique()} images -> {args.out}")
    print(f"locations len (pairs): mean {(lens.mean()/2):.2f} min {lens.min()//2} max {lens.max()//2}")


if __name__ == "__main__":
    main()
