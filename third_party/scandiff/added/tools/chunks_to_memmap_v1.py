"""Convert chunked ScanDiff DINO features to a verified fp16 memmap."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--images-list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stems = [line.rstrip("\n").split("\t")[1] for line in args.images_list.open()]
    if len(stems) != len(set(stems)):
        raise RuntimeError("duplicate stems in feature list")
    index = json.load((args.chunks / "index.json").open())
    missing = set(stems) - set(index)
    if missing:
        raise RuntimeError(f"missing {len(missing)} features")

    first_chunk = torch.load(args.chunks / index[stems[0]], map_location="cpu", weights_only=True)
    shape = tuple(first_chunk[stems[0]].shape)
    tmp_data = args.out / "features.npy.tmp"
    mm = np.memmap(tmp_data, dtype=np.float16, mode="w+", shape=(len(stems), *shape))
    out_index = {}
    cache_name = None
    cache = None
    # Write in chunk order. The cache below holds a single chunk, and the images
    # list is not ordered by chunk, so iterating it directly reloads a ~2 GB
    # chunk almost every image -- measured on the v2 set: 213,981 loads instead
    # of 210, i.e. hundreds of TB of reads instead of hundreds of GB. Row order
    # is irrelevant because every lookup goes through out_index.
    ordered = sorted(stems, key=lambda s: index[s])
    for row, stem in enumerate(ordered):
        chunk_name = index[stem]
        if chunk_name != cache_name:
            cache = torch.load(args.chunks / chunk_name, map_location="cpu", weights_only=True)
            cache_name = chunk_name
        mm[row] = cache[stem].numpy().astype(np.float16)
        out_index[stem] = row
        if row and row % 10000 == 0:
            print(f"{row}/{len(stems)}", flush=True)
    mm.flush()

    rng = random.Random(271828)
    # sample rows, then visit them in chunk order for the same reason as above
    sample = rng.sample(range(len(ordered)), min(100, len(ordered)))
    for row in sorted(sample, key=lambda r: index[ordered[r]]):
        stem = ordered[row]
        chunk_name = index[stem]
        if chunk_name != cache_name:
            cache = torch.load(args.chunks / chunk_name, map_location="cpu", weights_only=True)
            cache_name = chunk_name
        expected = cache[stem].numpy().astype(np.float16)
        if not np.array_equal(mm[row], expected):
            raise RuntimeError(f"verification failed for {stem}")

    tmp_index = args.out / "memmap_index.json.tmp"
    json.dump({"shape": [len(stems), *shape], "dtype": "float16", "index": out_index}, tmp_index.open("w"))
    tmp_data.rename(args.out / "features.npy")
    tmp_index.rename(args.out / "memmap_index.json")
    print(f"published {len(stems)} features at {args.out}")


if __name__ == "__main__":
    main()
