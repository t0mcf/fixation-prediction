"""plot training and validation metrics from a training log csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    if metric in train.columns and train[metric].notna().any():
        train_metric = pd.to_numeric(train[metric], errors="coerce")
        plt.plot(train["step"], train_metric, label=f"train {metric}", marker="o", markersize=3)

    if metric in val.columns and val[metric].notna().any():
        val_metric = pd.to_numeric(val[metric], errors="coerce")
        plt.plot(val["step"], val_metric, label=f"val {metric}", marker="o", markersize=5)

    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(f"{metric} over training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{metric}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"saved {output_path}", flush=True)


def main() -> None:
    args = parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    print("loaded log:", log_path, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", list(df.columns), flush=True)
    print(df.tail(), flush=True)

    for metric in ["kl", "nll", "nss"]:
        plot_metric(df, metric, output_dir)


if __name__ == "__main__":
    main()