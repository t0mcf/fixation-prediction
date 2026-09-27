"""
Build ScanDiff's MeanShift cluster file for a dataset in their format.
Adapted verbatim from src/data/components/mit1003/preprocess_MIT1003_clusters.py,
parametrized for an arbitrary {root}/{name}_fixations_{validation,test}.json layout.

Clusters are over validation+test scanpaths only (train excluded), scaled to 512x384.
Output: {root}/clusters_{dataset_name}_512_384.npy
Used by the evaluator for sequence_score / SS-Time metrics (NOT MultiMatch/ScanMatch).
"""
import argparse, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.cluster import MeanShift, estimate_bandwidth
warnings.filterwarnings("ignore")


def scanpath2clusters(meanshift, scanpath):
    string = []
    xs, ys = scanpath['X'], scanpath['Y']
    for i in range(len(xs)):
        string.append(meanshift.predict([[xs[i], ys[i]]])[0])
    return string


def improved_rate(meanshift, scanpaths):
    Nc = len(meanshift.cluster_centers_)
    Nb, Nw = 0, 0
    for scanpath in scanpaths:
        string = scanpath2clusters(meanshift, scanpath)
        for i in range(len(string) - 1):
            if string[i] == string[i + 1]:
                Nw += 1
            else:
                Nb += 1
    return (Nb - Nw) / Nc


def compute_clusters(gt_scanpaths):
    xs, ys = [], []
    for sp in gt_scanpaths:
        xs += list(sp['X']); ys += list(sp['Y'])
    gt_gaze = np.concatenate((np.vstack(xs), np.vstack(ys)), axis=1)
    bandwidth = estimate_bandwidth(gt_gaze)
    factors = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8]
    rates = []
    for factor in factors:
        bd = bandwidth * factor if bandwidth > 0.0 else None
        ms = MeanShift(bandwidth=bd); ms.fit(gt_gaze)
        rates.append(improved_rate(ms, gt_scanpaths))
    rates = np.vstack(rates)
    best_bd = factors[np.argmax(rates)] * bandwidth if bandwidth > 0.0 else None
    best_ms = MeanShift(bandwidth=best_bd); best_ms.fit(gt_gaze)
    gt_strings, subjects = [], []
    for sp in gt_scanpaths:
        gt_strings.append(scanpath2clusters(best_ms, sp))
        subjects.append(sp['subject'])
    return best_ms, gt_strings, subjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset dir (holds *_fixations_*.json)")
    ap.add_argument("--name", required=True, help="dataset name -> clusters_{name}_512_384.npy")
    ap.add_argument("--fix-prefix", default="mit1003",
                    help="fixation json filename prefix ({prefix}_fixations_{split}.json)")
    args = ap.parse_args()
    root = Path(args.root)
    TW, TH = 512, 384

    fixations = []
    for split in ("test", "validation"):
        fp = root / f"{args.fix_prefix}_fixations_{split}.json"
        fixations += json.load(open(fp, "rb"))

    data_dict = {}
    for sp in fixations:
        if sp['split'] == 'train':
            continue
        key = '{}-{}'.format(sp['split'], Path(sp['name']).stem)
        sp["X"] = (np.array(sp["X"]) / sp['width'] * TW).tolist()
        sp["Y"] = (np.array(sp["Y"]) / sp['height'] * TH).tolist()
        if len(sp["X"]) < 3:
            for _ in range(3 - len(sp["X"])):
                sp["X"].append(1); sp["Y"].append(1); sp["T"].append(1)
        data_dict.setdefault(key, []).append(sp)

    clusters = {}
    for key, value in tqdm(data_dict.items()):
        best_ms, gt_strings, subjects = compute_clusters(value)
        clusters[key] = {
            "strings": {k: v for k, v in zip(subjects, gt_strings)},
            "cluster": best_ms,
            "times": [s['T'][:len(s['X'])] for s in value],
        }

    out = root / f"clusters_{args.name}_512_384.npy"
    np.save(out, clusters, allow_pickle=True)
    print(f"saved {out}  ({len(clusters)} image-keys)")


if __name__ == "__main__":
    main()
