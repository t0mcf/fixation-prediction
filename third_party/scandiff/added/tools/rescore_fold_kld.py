"""Generalizes rescore_t90_kld.py to any fold/mode: compute the MultiMatch-KLD
+ ScanMatch-KLD metrics for an already-saved epoch-199 generation (fold 1-3
extension, results_catalog.md §8), without regenerating anything.
eval_mit1003_fold.sh never requested 'kld' in metrics_to_compute -- same
mistake the original eval_mit1003_t90.sh made, already fixed once for fold 0.

Usage:
  python tools/rescore_fold_kld.py --fold 1 --mode scratch
  python tools/rescore_fold_kld.py --fold 1 --mode ft
"""
import argparse
import sys
sys.path.insert(0, ".")
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf as OC

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--mode", choices=["scratch", "ft"], required=True)
args = ap.parse_args()

RUN = (f"/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_mit1003_fold{args.fold}"
       if args.mode == "scratch" else
       f"/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/ft_fold{args.fold}_av100k")

with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(config_name="eval.yaml", overrides=[
        f"data/train_datasets=[mit1003_fold{args.fold}]",
        f"data/val_datasets=[mit1003_fold{args.fold}]",
        f"data/test_datasets=[mit1003_fold{args.fold}]",
        f"ckpt_path={RUN}/train/checkpoints/last.pt",
    ])

from src.utils.instantiators import instantiate_datasets, instantiate_collators
from src.data.unified_datamodule import UnifiedDataModule

datasets = instantiate_datasets(cfg.get("data"))
collators = instantiate_collators(cfg.get("data"))
to_dict = OC.to_container(cfg.data)
for k in ["train_datasets", "val_datasets", "test_datasets",
          "train_collators", "val_collators", "test_collators"]:
    to_dict.pop(k, None)
datamodule = UnifiedDataModule(
    train_datasets=datasets["train_datasets"], val_datasets=datasets["val_datasets"],
    test_datasets=datasets["test_datasets"], train_collators=collators["train_collators"],
    val_collators=collators["val_collators"], test_collators=collators["test_collators"], **to_dict)
datamodule.setup()

from src.evaluator import Evaluator
ev = Evaluator(
    eval_root_path=f"{RUN}/eval_heldout",
    metrics_to_compute=["multi_match", "scan_match_no_dur", "kld"],
    data_to_extract=["preds", "metrics"],
    datamodule=datamodule,
)
metrics = ev.compute_metrics(epoch=199, is_validation=False)
print(f"===== FULL METRICS fold{args.fold} {args.mode} (incl. KLD) =====")
print(metrics)
