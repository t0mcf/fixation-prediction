"""
Extract DINOv2 features in ScanDiff's exact recipe, from an explicit image list.

Replicates src/data/components/mit1003/feature_extraction.py:dinoV2_image_data:
  model  = timm 'vit_base_patch14_reg4_dinov2.lvd142m' (pretrained, num_classes=0)
  input  = PIL.resize((518,518)) -> timm model transform
  feats  = model.forward_features(x)[:, 5:, :]   # drop 1 CLS + 4 register tokens
  save   -> {out_dir}/{stem}.pth   (shape: 1369 x 768)

Reads images_list.txt with lines "<abs_src_path>\t<stem>".
Runs on GPU if available (their original ran on CPU; identical math).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import PIL.Image
import torch
import timm
import torchvision.transforms as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-list", required=True)
    ap.add_argument(
        "--images-root",
        help="ImageNet train_images root; allows images-list to contain bare stems.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="for quick testing")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", help="store features as float16 (halves disk)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--av-center-crop", action="store_true",
        help="Apply Resize(256)->CenterCrop(224) before the square DINO input resize.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for line in open(args.images_list):
        parts = line.rstrip("\n").split("\t")
        if not parts or not parts[0]:
            continue
        if len(parts) >= 2:
            pairs.append(parts[:2])
        else:
            if not args.images_root:
                raise ValueError("Bare stems require --images-root")
            stem = parts[0]
            src = Path(args.images_root) / stem.split("_")[0] / f"{stem}.JPEG"
            pairs.append((str(src), stem))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"device={args.device} | {len(pairs)} images -> {out}", flush=True)

    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval().to(args.device)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    geometry_tf = T.Compose([T.Resize(256), T.CenterCrop(224)]) if args.av_center_crop else None

    # skip already-done, then process the rest in batches
    todo = [(s, st) for s, st in pairs if args.overwrite or not (out / f"{st}.pth").exists()]
    skipped = len(pairs) - len(todo)
    done = 0
    save_dtype = torch.float16 if args.fp16 else torch.float32
    first = True
    for b0 in range(0, len(todo), args.batch_size):
        chunk = todo[b0:b0 + args.batch_size]
        xs, stems = [], []
        for src, stem in chunk:
            try:
                img = PIL.Image.open(src).convert("RGB")
                if geometry_tf is not None:
                    img = geometry_tf(img)
                img = img.resize((518, 518))
            except Exception as e:
                print(f"  WARN open failed {src}: {e}", flush=True); continue
            xs.append(tf(img)); stems.append(stem)
        if not xs:
            continue
        x = torch.stack(xs).to(args.device)
        with torch.no_grad():
            feats = model.forward_features(x)[:, 5:, :].to(save_dtype).cpu()  # (b, 1369, 768)
        for j, stem in enumerate(stems):
            torch.save(feats[j].clone(), out / f"{stem}.pth")
        done += len(stems)
        if first:
            print(f"  feature shape: {tuple(feats[0].shape)} dtype={feats.dtype} "
                  f"(~{feats[0].numel()*feats.element_size()/1e6:.1f} MB/img)", flush=True)
            first = False
        if done % (args.batch_size * 10) < args.batch_size:
            print(f"  {done}/{len(todo)} (skipped {skipped})", flush=True)

    print(f"finished: wrote {done}, skipped {skipped} existing", flush=True)


if __name__ == "__main__":
    main()
