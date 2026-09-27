"""Precompute DINOv2 features for ImageNet images.

Saves features in chunk files (one .pth per --chunk-size images) to minimize
filesystem metadata operations on networked storage like Lustre.

Each chunk is a dict: {stem: tensor} where tensor is (1369, 768) fp16/fp32.
An index file (index.json) maps stem -> chunk filename for fast lookup.

Usage:
    python tools/precompute_features.py \
        --stems-file ./data/oursynth_200k/image_stems.txt \
        --images-dir /path/to/ImageNet/train_images \
        --out-dir /path/to/features \
        --batch-size 64 --fp16
"""

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ImageListDataset(Dataset):
    def __init__(self, image_paths, transform, geometry_transform=None):
        self.paths = image_paths
        self.transform = transform
        self.geometry_transform = geometry_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.geometry_transform is not None:
            img = self.geometry_transform(img)
        img = img.resize((518, 518))
        return self.transform(img), path.stem


def resolve_path(images_dir, img_filename):
    stem = Path(img_filename).stem
    class_dir = stem.rsplit("_", 1)[0]
    for ext in [".JPEG", ".jpeg", ".jpg", ".png"]:
        candidate = images_dir / class_dir / (stem + ext)
        if candidate.exists():
            return candidate
    direct = images_dir / img_filename
    if direct.exists():
        return direct
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", nargs="+", default=None,
                        help="fixation JSON files to extract image names from")
    parser.add_argument("--stems-file", default=None,
                        help="text file with one image stem per line")
    parser.add_argument("--images-list", default=None,
                        help="TSV with absolute image path and stem")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="save features in float16")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="images per chunk file (default: 1000)")
    parser.add_argument("--av-center-crop", action="store_true",
                        help="Apply Resize(256) and CenterCrop(224) before DINO input resize")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_paths = None
    if args.images_list:
        explicit_paths = []
        for line in open(args.images_list):
            source, stem = line.rstrip("\n").split("\t")[:2]
            explicit_paths.append((Path(source), stem))

    # collect unique image filenames
    all_names = set()
    if args.stems_file:
        for line in open(args.stems_file):
            stem = line.strip()
            if stem:
                all_names.add(f"{stem}.jpeg")
    if args.json_file:
        for jf in args.json_file:
            data = json.load(open(jf))
            for entry in data:
                all_names.add(entry.get("img_filename", entry.get("name")))
    if explicit_paths:
        all_names.update(f"{stem}.jpeg" for _, stem in explicit_paths)
    if not all_names:
        raise ValueError("Provide --json-file, --stems-file, or --images-list")
    print(f"Unique images needed: {len(all_names)}", flush=True)

    # load existing index to skip already-extracted stems
    index_path = out_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {}

    already = set(index.keys())
    todo_names = [n for n in all_names if Path(n).stem not in already]
    print(f"Already extracted: {len(already)}, remaining: {len(todo_names)}", flush=True)

    if not todo_names:
        print("Nothing to do.")
        return

    # resolve paths
    todo_paths = []
    missing = 0
    explicit_by_stem = {stem: source for source, stem in (explicit_paths or [])}
    for name in todo_names:
        stem = Path(name).stem
        p = explicit_by_stem.get(stem) or resolve_path(images_dir, name)
        if p is not None:
            todo_paths.append(p)
        else:
            missing += 1
    if missing:
        print(f"WARNING: {missing} images not found, skipping them", flush=True)
    print(f"Extracting features for {len(todo_paths)} images", flush=True)

    # find next chunk number
    existing_chunks = list(out_dir.glob("chunk_*.pth"))
    if existing_chunks:
        max_chunk = max(int(p.stem.split("_")[1]) for p in existing_chunks)
        next_chunk = max_chunk + 1
    else:
        next_chunk = 0

    # model
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,
    ).eval().cuda()
    for p in model.parameters():
        p.requires_grad = False

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    geometry_transform = T.Compose([T.Resize(256), T.CenterCrop(224)]) if args.av_center_crop else None
    dataset = ImageListDataset(todo_paths, transform, geometry_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    t0 = time.perf_counter()
    count = 0
    chunk_buf = {}

    def save_chunk():
        nonlocal next_chunk, chunk_buf
        if not chunk_buf:
            return
        chunk_name = f"chunk_{next_chunk:04d}.pth"
        torch.save(chunk_buf, out_dir / chunk_name)
        for stem in chunk_buf:
            index[stem] = chunk_name
        # update index on disk after each chunk
        with open(index_path, "w") as f:
            json.dump(index, f)
        next_chunk += 1
        chunk_buf = {}

    with torch.no_grad():
        for imgs, stems in loader:
            feats = model.forward_features(imgs.cuda())
            feats = feats[:, 5:, :]  # remove CLS + 4 register tokens
            feats = feats.cpu()
            if args.fp16:
                feats = feats.half()
            for i, stem in enumerate(stems):
                chunk_buf[stem] = feats[i]
            count += len(stems)

            if len(chunk_buf) >= args.chunk_size:
                save_chunk()

            if count % 1000 == 0 or count == len(todo_paths):
                elapsed = time.perf_counter() - t0
                rate = count / elapsed
                eta = (len(todo_paths) - count) / rate if rate > 0 else 0
                print(f"  {count}/{len(todo_paths)}  ({rate:.1f} img/s, ETA {eta/60:.0f} min)", flush=True)

    save_chunk()  # flush remaining

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {count} images in {elapsed/60:.1f} min ({count/elapsed:.1f} img/s)", flush=True)
    print(f"Chunks: {next_chunk} files in {out_dir}", flush=True)
    print(f"Index: {len(index)} entries", flush=True)


if __name__ == "__main__":
    main()
