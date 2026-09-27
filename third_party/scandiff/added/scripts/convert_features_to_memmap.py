"""Convert chunked .pth DINOv2 features to a memory-mapped numpy file.

Creates:
  features.npy  — (N, 1369, 768) float16 memmap
  memmap_index.json — {"stem": row_index, ...}

Usage:
  python scripts/convert_features_to_memmap.py data/oursynth_1k/dinov2_base_timm_image_features
  python scripts/convert_features_to_memmap.py data/oursynth_200k/dinov2_base_timm_image_features --stems-json data/oursynth_200k/oursynth_200k_fixations_train.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def get_needed_stems(fixation_jsons: list[str] | None) -> set[str] | None:
    """Extract unique image stems from fixation JSON files."""
    if not fixation_jsons:
        return None
    stems = set()
    for path in fixation_jsons:
        with open(path) as f:
            data = json.load(f)
        for sample in data:
            stems.add(Path(sample["name"]).stem)
    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="Path to dinov2_base_timm_image_features/")
    parser.add_argument("--stems-json", nargs="*",
                        help="Fixation JSON files to extract needed stems from. "
                             "If omitted, converts all images in the index.")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    index_path = features_dir / "index.json"

    if not index_path.exists():
        print("No index.json found — not a chunked feature directory.")
        return

    with open(index_path) as f:
        index = json.load(f)

    needed_stems = get_needed_stems(args.stems_json)
    if needed_stems:
        stems_to_convert = sorted(s for s in index if s in needed_stems)
        print(f"Converting {len(stems_to_convert)}/{len(index)} images (filtered by fixation JSONs)")
    else:
        stems_to_convert = sorted(index.keys())
        print(f"Converting all {len(stems_to_convert)} images")

    # Probe shape from first chunk
    first_chunk_name = index[stems_to_convert[0]]
    first_chunk = torch.load(features_dir / first_chunk_name, map_location="cpu", weights_only=True)
    sample_feat = first_chunk[stems_to_convert[0]]
    feat_shape = tuple(sample_feat.shape)
    feat_dtype = np.float16
    print(f"Feature shape: {feat_shape}, dtype: {feat_dtype}")

    N = len(stems_to_convert)
    out_path = features_dir / "features.npy"
    memmap = np.memmap(out_path, dtype=feat_dtype, mode="w+", shape=(N, *feat_shape))

    stem_to_idx = {}
    chunk_cache_name = None
    chunk_cache_data = None

    # Sort stems by chunk so we read each chunk file at most once
    stems_by_chunk = sorted(stems_to_convert, key=lambda s: index[s])

    for i, stem in enumerate(tqdm(stems_by_chunk, desc="Converting")):
        chunk_name = index[stem]
        if chunk_name != chunk_cache_name:
            chunk_cache_data = torch.load(features_dir / chunk_name, map_location="cpu", weights_only=True)
            chunk_cache_name = chunk_name
        feat = chunk_cache_data[stem]
        memmap[i] = feat.numpy().astype(feat_dtype)
        stem_to_idx[stem] = i

    memmap.flush()

    idx_path = features_dir / "memmap_index.json"
    with open(idx_path, "w") as f:
        json.dump({"shape": [N, *feat_shape], "dtype": "float16", "index": stem_to_idx}, f)

    print(f"Written: {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)")
    print(f"Written: {idx_path}")


if __name__ == "__main__":
    main()
