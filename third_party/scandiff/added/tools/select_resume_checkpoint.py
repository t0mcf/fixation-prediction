"""Print the newest loadable ScanDiff checkpoint, preferring last.pt."""
import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir", type=Path)
args = parser.parse_args()

candidates = [args.checkpoint_dir / "last.pt"]
candidates += sorted(
    args.checkpoint_dir.glob("epoch_*.pt"),
    key=lambda path: int(path.stem.split("_")[-1]),
    reverse=True,
)
for candidate in candidates:
    if not candidate.exists():
        continue
    try:
        checkpoint = torch.load(candidate, map_location="cpu")
        required = {"model", "optimizer", "global_step", "global_epoch"}
        if required <= checkpoint.keys():
            print(candidate)
            break
    except Exception as exc:
        print(f"Skipping invalid checkpoint {candidate}: {exc}", file=__import__("sys").stderr)
else:
    raise SystemExit("No loadable checkpoint found")
