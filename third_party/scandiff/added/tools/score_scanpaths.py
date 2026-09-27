"""
Score generated scanpaths against human MIT1003 scanpaths with ScanDiff's OWN
MultiMatch + KLD code, so the number is directly comparable to their table.

Inputs:
  --human  : human spec json {stem: {"size":[W,H], "scanpaths":[{X,Y,T}]}}
  --generated : np.save dict {stem: [ (L,2) arrays in [-1,1] ]}  (from sample_for_eval.py)

Reported "MultiMatch" follows their protocol (run_metrics.py):
  per MM dimension d, KLD( human-human score distribution || human-model score distribution );
  averaged over the 5 dims. LOWER = our scanpath statistics match humans better.
Also prints raw mean MM similarity (model-vs-human) and the human-vs-human ceiling.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.gazetools.metrics.multimatch import (
    compute_multi_match_score, compute_self_multi_match_score)
from src.gazetools.metrics.scanmatch import scan_match_score, self_scan_match_score
from src.gazetools.metrics.kld import perform_kld

# GazeParser 0.11.1 has a case bug: ScanMatch.__init__ calls self.CreateSubMatrix /
# self.GridMask but the methods are createSubMatrix / gridMask. Alias them.
import GazeParser.ScanMatch as _gsm
_gsm.ScanMatch.CreateSubMatrix = _gsm.ScanMatch.createSubMatrix
_gsm.ScanMatch.GridMask = _gsm.ScanMatch.gridMask

DIMS = ["Vector(Sh)", "Direction", "Length", "Position", "Duration"]


def scan_match_score_corrected(real: dict, generated: dict | None, tempbin=0):
    """ScanMatch with the width/height convention CORRECTED (2026-09-03).

    Upstream src/gazetools/metrics/scanmatch.py unpacks size as
    (imw, imh) = size[1], size[0] although every producer of 'size' (the
    authors' datasets and gen_to_dict above) stores [width, height]; on
    non-square images the 16x12 grid is therefore laid over the transposed
    frame and x coordinates are binned against the height. This variant uses
    Xres=width, Yres=height and the same fixed 16x12 grid (the upstream
    Xbin/Ybin 'adaptive' values are computed but never used there either).
    Identical to upstream on square images; reported alongside it, never as
    a silent replacement — paper-comparable rows keep the upstream numbers.
    generated=None gives the self (human-vs-human) scores.
    """
    from src.gazetools.metrics.scanmatch import _scan_match_score, ScanMatch
    scores = []
    for key, r in real.items():
        W, H = r["size"]
        try:
            m = ScanMatch(Xres=W, Yres=H, Xbin=16, Ybin=12, TempBin=tempbin)
        except Exception:
            print(f"Skipping image {key} (ScanMatch init)")
            continue
        g = generated[key]["scanpaths"] if generated is not None else None
        scores += _scan_match_score(m, r["scanpaths"], g)
    return scores


def gen_to_dict(generated_npy: dict, human: dict) -> dict:
    """Convert {stem: [(L,2) in [-1,1]]} -> {stem: {size, scanpaths:[{X,Y,T}]}} in pixels."""
    out = {}
    for stem, sps in generated_npy.items():
        if stem not in human:
            continue
        W, H = human[stem]["size"]
        scan = []
        for sp in sps:
            x = (np.asarray(sp)[:, 0] + 1) / 2 * (W - 1)
            y = (np.asarray(sp)[:, 1] + 1) / 2 * (H - 1)
            scan.append({"X": x.tolist(), "Y": y.tolist(), "T": [100.0] * len(x)})
        out[stem] = {"size": human[stem]["size"], "scanpaths": scan}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--label", default="ours")
    ap.add_argument("--n-images", type=int, default=None,
                    help="#training images for this checkpoint (x-axis of the scaling curve)")
    ap.add_argument("--temperature", type=float, default=None, help="sampling T (for sweep CSV)")
    ap.add_argument("--csv", default=None, help="append a results row to this CSV")
    args = ap.parse_args()

    human = json.load(open(args.human))
    generated_npy = np.load(args.generated, allow_pickle=True).item()
    generated = gen_to_dict(generated_npy, human)

    # restrict humans to stimuli we have generations for (apples-to-apples)
    human = {k: v for k, v in human.items() if k in generated}
    print(f"scoring {len(human)} images "
          f"(human sp: {sum(len(v['scanpaths']) for v in human.values())}, "
          f"model sp: {sum(len(v['scanpaths']) for v in generated.values())})")

    # per-image size handled via the 'size' key; globals are unused fallbacks
    g_mm = compute_multi_match_score(human, generated, 1, 1)   # human-vs-model pairs
    h_mm = compute_self_multi_match_score(human, 1, 1)         # human-vs-human pairs

    g_mm = [m for m in g_mm if not np.any(np.isnan(m))]
    h_mm = [m for m in h_mm if not np.any(np.isnan(m))]

    print(f"\n{'dim':<12}{'KLD(h||m)↓':>12}{'model sim':>12}{'human sim':>12}")
    klds = []
    for d in range(5):
        hd = [x[d] for x in h_mm]
        gd = [x[d] for x in g_mm]
        k = perform_kld(hd, gd)
        klds.append(k)
        print(f"{DIMS[d]:<12}{k:>12.4f}{np.mean(gd):>12.4f}{np.mean(hd):>12.4f}")
    # 5-dim average (incl. duration) and duration-free 4-dim average.
    # Our model predicts NO durations -> the duration-free 4-dim is the comparable number.
    mm_kld = float(np.mean(klds))
    mm_kld_nodur = float(np.mean(klds[:4]))
    g_sim = float(np.mean([np.mean(x) for x in g_mm]))
    h_sim = float(np.mean([np.mean(x) for x in h_mm]))
    g_sim_nd = float(np.mean([np.mean(x[:4]) for x in g_mm]))
    h_sim_nd = float(np.mean([np.mean(x[:4]) for x in h_mm]))
    print(f"{'-'*48}")
    print(f"{'MM 5-dim':<12}{mm_kld:>12.4f}{g_sim:>12.4f}{h_sim:>12.4f}")
    print(f"{'MM 4-dim*':<12}{mm_kld_nodur:>12.4f}{g_sim_nd:>12.4f}{h_sim_nd:>12.4f}")
    print(f"  (*duration-free: Vector/Direction/Length/Position — the comparable metric)")
    print(f"\n=> [{args.label}] MultiMatch duration-free (mean KLD, lower=better) = {mm_kld_nodur:.4f}")
    print(f"   raw MM similarity (4-dim): model-vs-human {g_sim_nd:.4f} | "
          f"human-vs-human ceiling {h_sim_nd:.4f}")

    # ----- ScanMatch (duration-free: tempbin=0), same protocol as their table -----
    print("\ncomputing ScanMatch (no-dur)...")
    g_sm = scan_match_score(real=human, generated=generated,
                            stimulus_width=1, stimulus_height=1, tempbin=0)
    h_sm = self_scan_match_score(data=human, stimulus_width=1, stimulus_height=1, tempbin=0)
    g_sm = [s for s in g_sm if s is not None and not np.isnan(s)]
    h_sm = [s for s in h_sm if s is not None and not np.isnan(s)]
    sm_kld = perform_kld(h_sm, g_sm)
    sm_model, sm_human = float(np.mean(g_sm)), float(np.mean(h_sm))
    print(f"=> [{args.label}] ScanMatch no-dur: KLD(h||m)↓ = {sm_kld:.4f} | "
          f"model {sm_model:.4f} | human ceiling {sm_human:.4f}")

    # ----- ScanMatch, width/height-corrected variant (== upstream on square images) -----
    g_smc = [s for s in scan_match_score_corrected(human, generated, tempbin=0)
             if s is not None and not np.isnan(s)]
    h_smc = [s for s in scan_match_score_corrected(human, None, tempbin=0)
             if s is not None and not np.isnan(s)]
    sm_kld_corr = perform_kld(h_smc, g_smc)
    print(f"=> [{args.label}] ScanMatch no-dur (W/H-corrected): KLD = {sm_kld_corr:.4f} | "
          f"model {np.mean(g_smc):.4f} | human ceiling {np.mean(h_smc):.4f}"
          f"{'   (square images: must equal upstream)' if all(v['size'][0]==v['size'][1] for v in human.values()) else ''}")

    if args.csv:
        import csv as _csv
        from pathlib import Path as _P
        row = {
            "label": args.label, "n_images": args.n_images, "temperature": args.temperature,
            "n_eval_images": len(human),
            "mm_kld_nodur": round(mm_kld_nodur, 5), "mm_sim_model": round(g_sim_nd, 5),
            "mm_sim_human": round(h_sim_nd, 5),
            "sm_kld": round(sm_kld, 5), "sm_model": round(sm_model, 5),
            "sm_human": round(sm_human, 5),
            "sm_kld_corr": round(sm_kld_corr, 5),
        }
        p = _P(args.csv)
        write_header = not p.exists()
        with p.open("a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"appended row -> {args.csv}")


if __name__ == "__main__":
    main()
