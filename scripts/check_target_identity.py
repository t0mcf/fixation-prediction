#!/usr/bin/env python
"""Gate: ours and DG3 were scored at the SAME target cells, for every
required run and every required grid. Fails on ANY missing evidence.

scripts/eval_synthetic_fair.py writes provenance.target_cells_sha256[<grid>]
= sha256 over the sorted (image, path id, step, row, col) tuples. The
digest is a property of the split, not the model, so it must be identical
across all ours seeds and DG3 rungs on the same split.

  python scripts/check_target_identity.py runs/synthetic_fair/validation
  python scripts/check_target_identity.py <dir> --require ours_1k_seed42 dg3_1k
  python scripts/check_target_identity.py <dir> --manifest <json>   # names from a frozen manifest

Exit 0 only if: every required run has a metrics.json with a digest for
every required grid, both families (ours_* and dg3_*) are present, and all
digests per grid are identical.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SIZES = ["1k", "10k", "50k", "100k", "200k", "400k", "800k"]
DEFAULT_OURS = ([f"ours_{s}_seed42" for s in SIZES]
                + [f"ours_{s}_seed43" for s in ["1k", "10k", "50k", "100k", "200k"]]
                + [f"ours_{s}_seed44" for s in ["1k", "10k", "50k"]])
DEFAULT_DG3 = [f"dg3_{s}" for s in ["1k", "10k", "50k", "100k", "200k"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="runs/synthetic_fair/validation")
    ap.add_argument("--require", nargs="+", default=None,
                    help="run names that must be present (default: the 15 ours + 5 DG3 runs)")
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest json; its record ids/names become the required set")
    ap.add_argument("--grids", nargs="+", default=["64", "224"])
    args = ap.parse_args()

    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        recs = m["records"] if isinstance(m, dict) else m
        required = [r.get("id") or r.get("name") for r in (recs if isinstance(recs, list) else recs.values())]
    else:
        required = args.require or (DEFAULT_OURS + DEFAULT_DG3)
    d = Path(args.dir)
    problems = []
    digests = defaultdict(dict)     # grid -> name -> digest
    for name in required:
        f = d / name / "metrics.json"
        if not f.exists():
            problems.append(f"missing result: {f}")
            continue
        m = json.loads(f.read_text())
        dig = m.get("provenance", {}).get("target_cells_sha256") or {}
        for g in args.grids:
            if g not in dig:
                problems.append(f"{name}: no target digest for grid {g}")
            else:
                digests[g][name] = dig[g]
    fams = {n.split("_")[0] for n in required}
    if not {"ours", "dg3"} <= fams:
        problems.append(f"cross-model check needs both families in the required set, got {sorted(fams)}")
    for g in args.grids:
        groups = defaultdict(list)
        for n, h in digests[g].items():
            groups[h].append(n)
        print(f"grid {g}: {len(digests[g])}/{len(required)} digests, {len(groups)} distinct")
        for h, names in groups.items():
            print(f"  {h[:16]}…  {', '.join(sorted(names))}")
        if len(groups) > 1:
            problems.append(f"grid {g}: target cells differ between runs")
    for p in problems:
        print("FAIL:", p)
    print("TARGET IDENTITY:", "OK" if not problems else f"NOT ESTABLISHED ({len(problems)} problem(s))")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
