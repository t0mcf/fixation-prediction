#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
EVAL = ROOT / "data/eval"
OUT = EVAL / "fixed_sequence_eval"
PY = "/mnt/vast-nhr/projects/nim00018/tom/envs/scandiff/bin/python"
COMPUTE = ROOT / "tools/compute_fixed_sequence_scores.py"
PLOT = ROOT / "tools/plot_fixed_sequence_metrics.py"


def add(entries, domain, model, scale, human, generated):
    slug = f"{domain}_{model.lower().replace(' ', '_')}_{scale}"
    entries.append({
        "domain": domain,
        "model": model,
        "scale": scale,
        "label": slug,
        "human": str(human),
        "generated": str(generated),
        "scores": str(OUT / "scores" / f"{slug}.pkl"),
    })


def build_manifest():
    entries = []
    mit = EVAL / "mit1003_human_m100.json"
    canon = EVAL / "canonical_val/canonical_val_gt.json"

    for scale in ("10k", "50k", "200k", "400k", "800k"):
        generated = EVAL / (f"gen_ours_{scale}_m100.npy" if scale in ("10k", "800k")
                            else f"gen_mit1003_{scale}.npy")
        add(entries, "zeroshot", "ours", int(scale[:-1]) * 1000, mit, generated)
    for scale in ("10k", "50k", "200k"):
        add(entries, "zeroshot", "DeepGaze III", int(scale[:-1]) * 1000, mit,
            EVAL / f"gen_mit1003_dg3_{scale}.npy")
    for source, scale in (("1k", 1000), ("10k_v2", 10000), ("50k", 50000),
                          ("100k", 100000), ("200k", 200000)):
        add(entries, "zeroshot", "ScanDiff", scale, mit,
            EVAL / f"gen_scandiff_{source}_m100_lenmatch.npy")

    for scale in ("10k", "50k", "200k", "400k", "800k"):
        add(entries, "indomain", "ours", int(scale[:-1]) * 1000, canon,
            EVAL / f"canonical_val/gen_cv_ours_{scale}_T1.0.npy")
    for scale in ("10k", "50k", "200k"):
        add(entries, "indomain", "DeepGaze III", int(scale[:-1]) * 1000, canon,
            EVAL / f"canonical_val/gen_cv_dg3_{scale}.npy")
    for scale in ("1k", "10k", "50k", "100k", "200k"):
        add(entries, "indomain", "ScanDiff", int(scale[:-1]) * 1000, canon,
            EVAL / f"canonical_val/gen_cv_scandiff_{scale}.npy")
    return entries


def compute(entry):
    out = Path(entry["scores"])
    if out.exists():
        return f"cached {entry['label']}"
    cmd = [PY, str(COMPUTE), "--human", entry["human"], "--generated", entry["generated"],
           "--out", entry["scores"]]
    subprocess.run(cmd, check=True)
    return f"done {entry['label']}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = build_manifest()
    missing = [e["generated"] for e in entries if not Path(e["generated"]).exists()]
    if missing:
        raise SystemExit("missing generations:\n" + "\n".join(missing))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(compute, e) for e in entries]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))
    subprocess.run([PY, str(PLOT), "--manifest", str(manifest), "--out-dir", str(OUT),
                    "--bootstraps", "500"], check=True)


if __name__ == "__main__":
    main()
