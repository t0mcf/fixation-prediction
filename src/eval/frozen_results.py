"""Full-precision, image-keyed results for the frozen MIT1003 evaluations."""
import json
import math
from pathlib import Path


def validate_split_files(split_dir):
    groups = {}
    for split, expected in (("train", 702), ("validation", 150), ("test", 151)):
        names = [Path(s.strip()).name.lower() for s in
                 (Path(split_dir) / (split + ".txt")).read_text().splitlines() if s.strip()]
        if len(names) != expected or len(set(names)) != expected:
            raise ValueError("Wrong count or duplicate names in " + split)
        groups[split] = set(names)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[a] & groups[b]:
            raise ValueError("Split overlap: " + a + "/" + b)
    return groups


def image_results(accumulator, nss_auc):
    rows = []
    for image_id, (ll, cb, count) in sorted(accumulator._by_image.items()):
        nss, auc, n = nss_auc[image_id]
        if n != count or count <= 0:
            raise ValueError("Metric fixation sets differ")
        rows.append(dict(image_id=image_id, n_fixations=int(count),
                         ll=(ll / count + accumulator._log_norm) / math.log(2),
                         ig=(ll - cb) / count / math.log(2),
                         nss=nss / count, auc=auc / count))
    return rows


def write_results(path, args, fold_results):
    # This export deliberately supports only the new fixed-split pathway.
    split_dir = getattr(args, "mit_split_dir", None) or getattr(args, "fixed_split_dir", None)
    if not split_dir or len(fold_results) != 1:
        raise ValueError("JSON export requires exactly one fixed split")
    expected = validate_split_files(split_dir)[args.eval_split]
    result = fold_results[0]
    for res, values in result.items():
        names = [row["image"] for row in values["per_image"]]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Incomplete or unexpected scored image set")
        if sum(row["n_fixations"] for row in values["per_image"]) != values["n_fixations"]:
            raise ValueError("Fixation count mismatch")
        for row in values["per_image"]:
            if not all(math.isfinite(row[k]) for k in ("ll", "ig", "nss", "auc")):
                raise ValueError("Nonfinite metric")
    payload = dict(schema_version=1, arguments=vars(args), results=result)
    # Exclusive creation: never overwrite a previous evaluation silently.
    with Path(path).open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
