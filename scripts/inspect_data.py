import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
print("[0] imports starting...", flush=True)

from src.data.dataset import ScanpathDataset
from src.data.dataloader import make_dataloader
print("[1] imports done", flush=True)


def tick(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe_tensor(name, x):
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}", flush=True)


def main():
    tick("creating dataset...")
    ds = ScanpathDataset(
        split="train",
        max_images=5,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )
    tick("dataset created")

    print("\ndataset", flush=True)
    print("len(ds):", len(ds), flush=True)
    print("unique images:", ds.df["image_path"].nunique(), flush=True)
    print("columns:", list(ds.df.columns), flush=True)

    tick("loading single sample ds[0]...")
    sample = ds[0]
    tick("loaded single sample")

    for k, v in sample.items():
        if hasattr(v, "shape"):
            describe_tensor(k, v)
        else:
            print(f"{k}: {v}", flush=True)

    print("prefix_len:", sample["prefix_len"], flush=True)
    print("target_xy:", sample["target_xy"], flush=True)
    print("heatmap sum:", float(sample["heatmap"].sum()), flush=True)
    print("heatmap max:", float(sample["heatmap"].max()), flush=True)

    print("\nrandom t behavior for same index", flush=True)
    for i in range(5):
        tick(f"loading ds[0], call {i}...")
        s = ds[0]
        print(
            f"call {i}: prefix_len={s['prefix_len']}, "
            f"target_xy={s['target_xy'].numpy()}",
            flush=True,
        )

    tick("creating dataloader...")
    loader = make_dataloader(
        split="train",
        batch_size=4,
        num_workers=0,
        max_images=5,
        use_grouped_sampler=False,
        heatmap_sigma=2.0,
        heatmap_size=64,
        parquet_path="data/debug_scanpaths.parquet",
    )
    tick("dataloader created")

    tick("loading first batch...")
    batch = next(iter(loader))
    tick("loaded first batch")

    print("\nbatch", flush=True)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            describe_tensor(k, v)
        else:
            print(f"{k}: {v}", flush=True)

    print("batch heatmap sums:", batch["heatmap"].sum(dim=(-1, -2)), flush=True)
    print("batch prefix_len:", batch["prefix_len"], flush=True)


if __name__ == "__main__":
    main()