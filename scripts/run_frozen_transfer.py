"""Run or check ONE manifest-selected evaluation. Never submits SLURM jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_validation(manifest, manifest_hash, outputs):
    reference_counts = None
    for rec in manifest["records"]:
        directory = outputs / "validation" / rec["id"]
        meta = json.loads((directory / "provenance.json").read_text())
        assert meta["manifest_sha256"] == manifest_hash and meta["status"] == "complete"
        assert meta["result_sha256"] == digest(directory / "metrics.json")
        result = json.loads((directory / "metrics.json").read_text())["results"]["224"]
        counts = {r["image"]: r["n_fixations"] for r in result["per_image"]}
        assert len(counts) == 150
        if reference_counts is None:
            reference_counts = counts
        assert counts == reference_counts, "Different per-image fixation sets: " + rec["id"]
        for short, full in (("ll", "ll_uniform_img"), ("nss", "nss_img"), ("auc", "auc_img")):
            mean = sum(r[short] for r in result["per_image"]) / 150
            assert abs(mean - result[full]) < 1e-6
    print("VALIDATION GATE PASSED: all 30 selected runs; identical per-image fixation counts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--index", type=int)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--check", action="store_true", help="Hashes/command only; no inference or outputs")
    ap.add_argument("--verify-validation", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    manifest_hash = digest(args.manifest)
    outputs = Path(args.outputs).resolve()
    for p, expected in manifest["dependencies"].items():
        if digest(p) != expected:
            raise RuntimeError("Frozen dependency changed: " + p)
    if args.verify_validation:
        verify_validation(manifest, manifest_hash, outputs)
        return
    if args.index is None or not 0 <= args.index < len(manifest["records"]):
        ap.error("index must identify a manifest row")
    rec = manifest["records"][args.index]
    if digest(rec["checkpoint"]) != rec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after selection")
    out = outputs / args.split / rec["id"]
    if out.exists():
        raise FileExistsError("Refusing to overwrite evaluation: " + str(out))
    if rec["family"] == "ours":
        cwd = ROOT
        command = [sys.executable, "-B", "scripts/eval_mit1003_fair.py", "--mode", "finetuned",
                   "--checkpoint", rec["checkpoint"], "--mit-split-dir", manifest["split_dir"]]
    else:
        cwd = ROOT.parent / "scanpather"
        command = [sys.executable, "-B", "-m", "scanpath.eval_mit1003_fair_dg3",
                   "--run-dir", rec["run"], "--stage", rec["stage"], "--ckpt-name", "best.pth",
                   "--twosize-dir", str(cwd / "runs/dg3_mit1003_ourfolds_scratch/MIT1003_twosize"),
                   "--fixed-split-dir", manifest["split_dir"]]
    command += ["--eval-split", args.split, "--resolutions", "224", "--name", rec["id"],
                "--output-json", str(out / "metrics.json")]
    if args.check:
        print(json.dumps(dict(id=rec["id"], cwd=str(cwd), command=command,
                              test_requires_validation_gate=True), indent=2))
        return
    if args.split == "test":
        verify_validation(manifest, manifest_hash, outputs)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution required; do not run full inference on a login node")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(manifest_sha256=manifest_hash, record=rec, command=command,
                    split=args.split, status="running", torch_version=torch.__version__,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = dict(os.environ, OURS_CROP_TRANSFORM="0", PYTHONDONTWRITEBYTECODE="1")
    with (out / "evaluation.log").open("x") as log:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("Evaluation failed; inspect " + str(out / "evaluation.log"))
    result = json.loads((out / "metrics.json").read_text())
    assert result["arguments"]["eval_split"] == args.split
    assert result["results"]["224"]["n_images"] == (150 if args.split == "validation" else 151)
    if digest(rec["checkpoint"]) != rec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    metadata.update(status="complete", result_sha256=digest(out / "metrics.json"))
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("completed", rec["id"], args.split, flush=True)


if __name__ == "__main__":
    main()
