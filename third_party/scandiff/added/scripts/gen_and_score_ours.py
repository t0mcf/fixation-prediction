"""Generate ScanDiff scanpaths on a small canonical-val subset and score them
with ScanDiff's own MultiMatch/ScanMatch implementations against ground truth.

Also dumps the ground-truth scanpath dict (already in ScanDiff's 512x384 pixel
convention) to disk so a second script can score our own model's scanpaths
against the *exact same* reference set with the *exact same* scoring code.

Usage:
  python scripts/gen_and_score_ours.py --size 1k --n-images 40 --respacing 100
  python scripts/gen_and_score_ours.py --size 10k --n-images 40 --respacing 100
"""
import argparse
import pickle
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra

import rootutils
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.create_diffusion import create_diffusion
from src.utils.instantiators import instantiate_datasets, instantiate_collators
from src.data.unified_datamodule import UnifiedDataModule
from src.evaluator import Evaluator
from src.utils.test_utils import process_data_for_metrics_computation
from src.gazetools.metrics.run_metrics import compute_multi_match_score, compute_self_multi_match_score
from src.gazetools.metrics.scanmatch import scan_match_score

CONFIG_DIR = str(ROOT / "configs")
IMG_W, IMG_H = 512, 384  # matches process_data_for_metrics_computation's hardcoded canvas


def build_cfg(size: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"data/val_datasets=[oursynth_{size}]",
                f"data/train_datasets=[oursynth_{size}]",
                f"data/test_datasets=[oursynth_{size}]",
                "trainer=gpu",
                "callbacks=default",
                "train=false",
                "test=false",
                "diffusion.num_timesteps=1000",
                "diffusion_class=spaced_diffusion",
                f"hydra.run.dir=./runs_ours/genscore_{size}",
            ],
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="1k or 10k")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--respacing", type=int, default=100, help="diffusion steps at sampling time (fewer=faster)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = ROOT / f"runs_ours/genscore_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.size)

    datasets = instantiate_datasets(cfg.get("data"))
    collators = instantiate_collators(cfg.get("data"))
    to_dict_config = OmegaConf.to_container(cfg.data)
    for k in ["train_datasets", "val_datasets", "test_datasets",
              "train_collators", "val_collators", "test_collators"]:
        to_dict_config.pop(k, None)
    to_dict_config["batch_size_validation"] = 1
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
    n_val = len(datamodule.data_val)
    print(f"val images available: {n_val}")

    # val_dataloader() always uses shuffle=False, and the underlying fixation
    # JSON is sorted by ImageNet class -- taking the first N rows means the
    # first N/subjects_per_image *classes*, not a representative sample.
    # Shuffle each wrapped dataset's row list in place (fixed seed) so a
    # sequential slice of val_dataloader spans the class distribution.
    import random
    for ds in datamodule.data_val.datasets:
        random.Random(3141).shuffle(ds.samples["sequences"])
    n_classes_in_slice = len({
        s["name"].split("_")[0]
        for s in datamodule.data_val.datasets[0].samples["sequences"][: args.n_images]
    })
    print(f"after shuffle: first {args.n_images} rows span {n_classes_in_slice} distinct classes")

    model = hydra.utils.instantiate(cfg.model).to(args.device)
    ckpt_path = ROOT / f"runs_ours/train_{args.size}/train/checkpoints/last.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint {ckpt_path} (epoch {ckpt.get('global_epoch')})")

    # respaced diffusion for faster sampling than the full 1000-step chain used in training
    diffusion = create_diffusion(
        cfg, timestep_respacing=str(args.respacing),
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )

    evaluator = hydra.utils.instantiate(
        cfg.evaluation, datamodule=datamodule, eval_root_path=str(out_dir), _target_=Evaluator
    )
    evaluator.limit_test_batches = min(1.0, args.n_images / n_val)
    print(f"limit_test_batches={evaluator.limit_test_batches:.4f} -> ~{args.n_images} images")

    preds, target = evaluator.extract_predictions(model, diffusion, epoch=0, is_validation=True)
    print(f"generated scanpaths for {len(preds)} images")

    gt = process_data_for_metrics_computation(datamodule, target, is_original=True)
    gen = process_data_for_metrics_computation(datamodule, preds, is_original=False)

    with open(out_dir / "ground_truth.pkl", "wb") as f:
        pickle.dump(gt, f)
    print(f"saved ground truth ({len(gt)} images) to {out_dir/'ground_truth.pkl'}")

    # compute_multi_match_score returns a flat list of 5-tuples (vector, direction,
    # length, position, duration), one per (ground-truth, generated) pair across all
    # images -- matches the aggregation in src/gazetools/metrics/run_metrics.py exactly.
    mm = compute_multi_match_score(gt, gen, IMG_W, IMG_H)
    mm_means = [sum(x) / len(x) for x in mm]
    print(f"ScanDiff MultiMatch (mean over {len(mm_means)} real/gen pairs): {sum(mm_means)/len(mm_means):.4f}")

    sm_scores = scan_match_score(real=gt, generated=gen, stimulus_width=IMG_W, stimulus_height=IMG_H, tempbin=0)
    print(f"ScanDiff ScanMatch (mean over {len(sm_scores)} pairs): {sum(sm_scores)/len(sm_scores):.4f}")

    with open(out_dir / "scandiff_results.pkl", "wb") as f:
        pickle.dump({"multimatch_per_image": mm_means, "scanmatch_scores": sm_scores,
                     "multimatch_mean": sum(mm_means)/len(mm_means),
                     "scanmatch_mean": sum(sm_scores)/len(sm_scores)}, f)
    print(f"saved results to {out_dir/'scandiff_results.pkl'}")


if __name__ == "__main__":
    main()
