"""Build ScanDiff JSON datasets from the frozen thesis_protocol/v1 splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_key(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[-2:])


def read_keys(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def entries(frame: pd.DataFrame, split: str, canvas: int) -> list[dict]:
    result = []
    for row in frame.itertuples(index=False):
        locations = np.asarray(row.locations, dtype=np.float64).reshape(-1, 2)
        n_fixations = len(np.asarray(row.log_pis)) + 1
        locations = locations[:n_fixations]
        if n_fixations < 3:
            continue
        xy = (locations + 1.0) * 0.5
        if np.any((xy < 0.0) | (xy > 1.0)):
            raise ValueError(f"out-of-frame fixation in {row.image_path}, epoch={row.epoch}")
        stem = Path(row.image_path).stem
        result.append({
            "name": f"{stem}.jpeg",
            "img_filename": f"{stem}.jpeg",
            "split": split,
            "width": canvas,
            "height": canvas,
            "X": (xy[:, 0] * canvas).tolist(),
            "Y": (xy[:, 1] * canvas).tolist(),
            "T": [200.0] * n_fixations,
            "subject": int(row.epoch),
        })
    return result


def select(frame: pd.DataFrame, keys: list[str], epochs: set[int]) -> pd.DataFrame:
    order = {key: index for index, key in enumerate(keys)}
    selected = frame[frame["key"].isin(order) & frame["epoch"].isin(epochs)].copy()
    selected["image_order"] = selected["key"].map(order)
    selected = selected.sort_values(["image_order", "epoch"]).drop(columns="image_order")
    counts = selected.groupby("key")["epoch"].nunique()
    bad = counts[counts != len(epochs)]
    missing = set(keys) - set(counts.index)
    if len(bad) or missing:
        raise RuntimeError(f"incomplete split: bad={len(bad)}, missing={len(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=["1k", "10k", "50k", "100k", "200k"])
    parser.add_argument("--canvas", type=int, default=1000)
    args = parser.parse_args()

    split_root = args.protocol / "splits" / "imagenet_av"
    train_epochs = {int(x) for x in read_keys(split_root / "scanpath_ids_train.txt")}
    eval_epochs = {int(x) for x in read_keys(split_root / "scanpath_ids_evaluation.txt")}
    validation_keys = read_keys(split_root / "validation_5k.txt")
    test_keys = read_keys(split_root / "test_10k.txt")

    frame = pd.read_parquet(args.parquet, engine="pyarrow")
    frame["key"] = frame["image_path"].map(canonical_key)
    if frame["key"].duplicated().all():
        raise RuntimeError("invalid canonical image keys")

    validation = select(frame, validation_keys, eval_epochs)
    test = select(frame, test_keys, eval_epochs)
    union_keys = set(validation_keys) | set(test_keys)

    for scale in args.scales:
        train_keys = read_keys(split_root / f"train_{scale}.txt")
        if union_keys & set(train_keys):
            raise RuntimeError(f"train/eval overlap at {scale}")
        train = select(frame, train_keys, train_epochs)
        out = args.out_root / f"oursynth_v1_{scale}"
        out.mkdir(parents=True, exist_ok=True)
        name = out.name
        payloads = {
            "train": entries(train, "train", args.canvas),
            "validation": entries(validation, "validation", args.canvas),
            "test": entries(test, "test", args.canvas),
        }
        for split, payload in payloads.items():
            with (out / f"{name}_fixations_{split}.json").open("w") as handle:
                json.dump(payload, handle)
        print(scale, {key: len(value) for key, value in payloads.items()}, flush=True)

    largest = read_keys(split_root / f"train_{args.scales[-1]}.txt")
    all_keys = largest + validation_keys + test_keys
    feature_list = args.out_root / "oursynth_v1_200k_feature_images.tsv"
    with feature_list.open("w") as handle:
        for key in all_keys:
            source = args.imagenet_root / key
            handle.write(f"{source}\t{source.stem}\n")
    print(f"feature list: {feature_list} ({len(all_keys)} images)", flush=True)


if __name__ == "__main__":
    main()
