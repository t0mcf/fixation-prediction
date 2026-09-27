"""One-off: compute the MultiMatch-KLD metric (paper convention) for the
train_mit1003_t90 checkpoint's already-saved epoch-199 generations, without
regenerating anything. The original eval_mit1003_t90.sh run never requested
'kld' in metrics_to_compute, so it was silently never computed."""
import sys
sys.path.insert(0, ".")
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf as OC

with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(config_name="eval.yaml", overrides=[
        "data/train_datasets=[mit1003_t90]",
        "data/val_datasets=[mit1003_t90]",
        "data/test_datasets=[mit1003_t90]",
        "ckpt_path=/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_mit1003_t90/train/checkpoints/last.pt",
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
    eval_root_path="/mnt/vast-nhr/projects/nim00018/tom/scandiff/runs_ours/train_mit1003_t90/eval_heldout",
    metrics_to_compute=["multi_match", "scan_match_no_dur", "kld"],
    data_to_extract=["preds", "metrics"],
    datamodule=datamodule,
)
metrics = ev.compute_metrics(epoch=199, is_validation=False)
print("===== FULL METRICS (incl. KLD, paper-comparable) =====")
print(metrics)
