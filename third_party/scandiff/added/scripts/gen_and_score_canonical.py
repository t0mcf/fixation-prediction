"""Generate ScanDiff scanpaths for the canonical val subset (built by
build_canonical_val_dataset.py, features by tools/extract_dinov2_features.py)
and score with ScanDiff's own MultiMatch/ScanMatch, exactly like
gen_and_score_ours.py but bypassing hydra's oursynth_{size} datasets entirely
since the canonical subset isn't part of that per-scale config system.

Usage:
  python scripts/gen_and_score_canonical.py --size 1k --respacing 100
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
from src.data.components.mit1003.mit1003 import MIT1003Dataset
from src.data.components.collator import Collator
from src.data.unified_datamodule import UnifiedDataModule
from src.evaluator import Evaluator
from src.utils.test_utils import process_data_for_metrics_computation
from src.gazetools.metrics.run_metrics import compute_multi_match_score, compute_self_multi_match_score
from src.gazetools.metrics.scanmatch import scan_match_score

CONFIG_DIR = str(ROOT / "configs")
IMG_W, IMG_H = 512, 384
CANON_DIR = ROOT / "runs_ours" / "canonical_val"


def build_cfg(size: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"data/val_datasets=[oursynth_{size}]",
                f"data/train_datasets=[oursynth_{size}]",
                f"data/test_datasets=[oursynth_{size}]",
                "trainer=gpu", "callbacks=default",
                "train=false", "test=false",
                "diffusion.num_timesteps=1000", "diffusion_class=spaced_diffusion",
                f"hydra.run.dir=./runs_ours/genscore_canonical_{size}",
            ],
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="1k or 10k -- selects which trained checkpoint to load")
    ap.add_argument("--respacing", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = ROOT / f"runs_ours/genscore_canonical_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.size)

    # root_path/{name}_fixations_{split}.json is what MIT1003Dataset expects;
    # our builder wrote plain fixations_validation.json -- symlink once so the
    # constructor's expected filename resolves, *before* instantiating.
    expected = CANON_DIR / "canonical_val_fixations_validation.json"
    if not expected.exists():
        expected.symlink_to(CANON_DIR / "fixations_validation.json")

    # canonical-val dataset, built by build_canonical_val_dataset.py, standing
    # in for the usual hydra oursynth_{size} val dataset.
    canon_dataset = MIT1003Dataset(
        name="canonical_val", root_path=str(CANON_DIR), task="What do you see in the image?",
        split="valid", num_subjects=16, time_in_ms=False, use_abs_coords=True,
        task_embeddings_file="task_embeddings.npy", img_features_dir="features",
    )
    print(f"canonical val dataset: {len(canon_dataset)} fixation rows")

    collator = Collator(max_len=16, use_abs_coords=True, time_in_ms=False)

    to_dict_config = OmegaConf.to_container(cfg.data)
    for k in ["train_datasets", "val_datasets", "test_datasets",
              "train_collators", "val_collators", "test_collators"]:
        to_dict_config.pop(k, None)
    to_dict_config["batch_size_validation"] = 1
    datamodule = UnifiedDataModule(
        train_datasets=[canon_dataset],
        val_datasets=[canon_dataset],
        test_datasets=[canon_dataset],
        train_collators=[collator],
        val_collators=[collator],
        test_collators=[collator],
        **to_dict_config,
    )
    datamodule.setup()
    n_val = len(datamodule.data_val)
    print(f"val rows available: {n_val}")

    model = hydra.utils.instantiate(cfg.model).to(args.device)
    ckpt_path = ROOT / f"runs_ours/train_{args.size}/train/checkpoints/last.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint {ckpt_path} (epoch {ckpt.get('global_epoch')})")

    diffusion = create_diffusion(
        cfg, timestep_respacing=str(args.respacing),
        diffusion_steps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )

    evaluator = hydra.utils.instantiate(
        cfg.evaluation, datamodule=datamodule, eval_root_path=str(out_dir), _target_=Evaluator
    )
    evaluator.limit_test_batches = 1.0  # canonical subset is already small (100 images x 16 = 1600 rows)

    preds, target = evaluator.extract_predictions(model, diffusion, epoch=0, is_validation=True)
    print(f"generated scanpaths for {len(preds)} images")

    gt = process_data_for_metrics_computation(datamodule, target, is_original=True)
    gen = process_data_for_metrics_computation(datamodule, preds, is_original=False)

    with open(out_dir / "ground_truth.pkl", "wb") as f:
        pickle.dump(gt, f)
    print(f"saved ground truth ({len(gt)} images) to {out_dir/'ground_truth.pkl'}")

    mm = compute_multi_match_score(gt, gen, IMG_W, IMG_H)
    mm_means = [sum(x) / len(x) for x in mm]
    print(f"ScanDiff MultiMatch (mean over {len(mm_means)} real/gen pairs): {sum(mm_means)/len(mm_means):.4f}")

    sm_scores = scan_match_score(real=gt, generated=gen, stimulus_width=IMG_W, stimulus_height=IMG_H, tempbin=0)
    print(f"ScanDiff ScanMatch (mean over {len(sm_scores)} pairs): {sum(sm_scores)/len(sm_scores):.4f}")

    human_mm = compute_self_multi_match_score(gt, IMG_W, IMG_H)
    human_mm_means = [sum(x) / len(x) for x in human_mm]
    print(f"Human (leave-one-out) MultiMatch: {sum(human_mm_means)/len(human_mm_means):.4f}")

    with open(out_dir / "scandiff_results.pkl", "wb") as f:
        pickle.dump({"multimatch_per_image": mm_means, "scanmatch_scores": sm_scores,
                     "multimatch_mean": sum(mm_means)/len(mm_means),
                     "scanmatch_mean": sum(sm_scores)/len(sm_scores),
                     "human_multimatch_mean": sum(human_mm_means)/len(human_mm_means)}, f)
    print(f"saved results to {out_dir/'scandiff_results.pkl'}")


if __name__ == "__main__":
    main()
