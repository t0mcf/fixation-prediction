"""
Build the AV-fixed-5 pretraining parquets for the SD-5 / AV-fixed-5-full /
AV-fixed-5-match generator triangle.

Holds path count and path length constant across the pretraining arms so the
comparison isolates the data source: AV is materialised as parquets with the
same per-image path count as the SD corpus, mirroring its schema so all arms
train through the identical path (--train-parquet-path, fixed sampler, same
geometry toggle).

Outputs (same schema as scandiffgen_100k: image_path, epoch, locations, log_pis):
  av_fixed5_full_100k/merged.parquet   5 real AV paths/image, natural length (16)
  av_fixed5_match_100k/merged.parquet  same 5, each truncated to the index-paired
                                       SDgen path's length (SD length distribution)

Geometry-agnostic: coordinates stay in the generator's [-1,1] frame exactly as the
SD parquet stores them; the crop-vs-resize decision is a train-time image transform
(OURS_CROP_TRANSFORM) applied uniformly to all three arms, decided LATER.

  python scripts/build_av_fixed5_parquets.py --validate-only   # checks, no write
  python scripts/build_av_fixed5_parquets.py                   # build + validate
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SD = "/mnt/vast-nhr/projects/nim00018/tom/scandiff/data/scandiffgen_100k/merged.parquet"
REAL = ("/mnt/vast-nhr/projects/nim00018/datasets/scanpath_parquet"
        "/2026_03_03_20_41_11_268d00ae/scanpaths/merged.parquet")
OUT_BASE = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff/data")
SP_PER_IMAGE = 5


def _valid_len(loc_flat) -> int:
    xy = np.asarray(loc_flat, float).reshape(-1, 2)
    return int((~np.isnan(xy[:, 0])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--sd-parquet", default=SD,
                    help="SD teacher corpus the arms are matched to. For the "
                         "v2 triangle: the UNFILTERED official-teacher corpus "
                         "(out-of-bounds sequences kept in).")
    ap.add_argument("--tag", default="100k",
                    help="suffix for the output dirs, e.g. v2u_100k -> "
                         "av_fixed5_{full,match}_v2u_100k")
    args = ap.parse_args()

    print(f"loading SD parquet {args.sd_parquet} ...", flush=True)
    sd = pd.read_parquet(args.sd_parquet, columns=["image_path", "epoch", "locations"])
    # SD image order = the exact image set/order the SD-5 arm trains on.
    sd_imgs = list(dict.fromkeys(sd["image_path"]))
    assert len(sd_imgs) == 100000, f"expected 100k SD images, got {len(sd_imgs)}"
    sd_g = sd.groupby("image_path", sort=False)

    # Per-image SD path lengths (index order within the image group). The
    # unfiltered v2 corpus has 3-5 paths per image (702 too-short slots carry
    # no prediction target and are absent); the AV arms take exactly as many
    # paths per image as SD has, so path COUNT stays matched per image.
    sd_lengths = {}
    for ip, grp in sd_g:
        lens = [_valid_len(l) for l in grp["locations"].tolist()]
        assert 1 <= len(lens) <= SP_PER_IMAGE, f"SD image {ip} has {len(lens)} paths"
        sd_lengths[ip] = lens

    print("loading REAL parquet (image-filtered) ...", flush=True)
    sd_img_set = set(sd_imgs)
    real = pd.read_parquet(REAL, columns=["image_path", "epoch", "locations", "log_pis"])
    real = real[real["image_path"].isin(sd_img_set)]
    missing = sd_img_set - set(real["image_path"].unique())
    assert not missing, f"{len(missing)} SD images missing from REAL parquet"
    real_g = real.groupby("image_path", sort=False)

    full_rows, match_rows = [], []
    for ip in sd_imgs:
        grp = real_g.get_group(ip)
        sd_lens = sd_lengths[ip]
        # deterministic head(n) where n = SD's per-image path count (usually 5)
        head = grp.head(len(sd_lens))
        assert len(head) == len(sd_lens), f"REAL image {ip} has <{len(sd_lens)} paths"
        for j, (_, r) in enumerate(head.iterrows()):
            loc = np.asarray(r["locations"], dtype=np.float32)
            lp = np.asarray(r["log_pis"], dtype=np.float32) if r["log_pis"] is not None else None
            xy = loc.reshape(-1, 2)
            avail = _valid_len(loc)
            # full: keep AV's natural (valid) fixations
            full_xy = xy[:avail]
            # match: truncate to the index-paired SD path length (never upsample)
            tlen = min(sd_lens[j], avail)
            match_xy = xy[:tlen]
            base = {"image_path": ip, "epoch": np.int16(r["epoch"])}
            full_rows.append({**base, "locations": full_xy.reshape(-1).astype(np.float32),
                              "log_pis": (lp[:avail] if lp is not None else None)})
            match_rows.append({**base, "locations": match_xy.reshape(-1).astype(np.float32),
                               "log_pis": (lp[:tlen] if lp is not None else None)})

    full_df = pd.DataFrame(full_rows)
    match_df = pd.DataFrame(match_rows)

    # ---- validation ----
    def summarize(name, df, ref_lens=None):
        spi = df.groupby("image_path", sort=False).size()
        lens = df["locations"].map(lambda a: len(a) // 2)
        print(f"\n[{name}] rows={len(df)} imgs={df['image_path'].nunique()} "
              f"sp/image[min/mean/max]={spi.min()}/{spi.mean():.2f}/{spi.max()}")
        print(f"    fix-len[min/mean/max]={lens.min()}/{lens.mean():.3f}/{lens.max()}")
        return lens

    sd_all_lens = pd.Series([l for ls in sd_lengths.values() for l in ls])
    print(f"\n[SD-5 ref] fix-len[min/mean/max]="
          f"{sd_all_lens.min()}/{sd_all_lens.mean():.3f}/{sd_all_lens.max()}")
    summarize("AV-fixed-5-full", full_df)
    match_lens = summarize("AV-fixed-5-match", match_df)

    # match distribution should track SD's (equal when AV avail(16) >= SD len)
    dl = abs(match_lens.mean() - sd_all_lens.mean())
    print(f"\nmean fix-len |AV-match - SD| = {dl:.4f} (should be ~0)")
    assert dl < 0.05, "AV-match length distribution does not track SD"
    assert set(full_df["image_path"]) == sd_img_set, "full image set mismatch"
    assert set(match_df["image_path"]) == sd_img_set, "match image set mismatch"
    # coordinate frame sanity: same [-1,1]-ish range as SD, untouched
    fx = np.concatenate(full_df["locations"].values)
    print(f"AV-full coord range: [{fx.min():.3f}, {fx.max():.3f}] (generator frame, untouched)")

    # ---- prefix-residual quantification ----
    # AV-match is a PREFIX truncation, so it carries AV's *early-scanpath*
    # statistics, not "AV shortened". These numbers quantify that residual for
    # the thesis ("isolates source from length up to a named residual").
    def geom(df, label):
        pts, saccs = [], []
        for a in df["locations"].values:
            xy = np.asarray(a, np.float32).reshape(-1, 2)
            if len(xy) >= 2:
                pts.append(xy)
                saccs.append(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        p = np.concatenate(pts); s = np.concatenate(saccs)
        print(f"    [{label:9s}] mean|r|={np.linalg.norm(p, axis=1).mean():.4f}  "
              f"x_sd={p[:,0].std():.4f}  mean sacc={s.mean():.4f}")
    print("\nprefix residual (AV frame [-1,1], first fixation included):")
    geom(full_df, "AV-full")
    geom(match_df, "AV-match")

    if args.validate_only:
        print("\n--validate-only: no parquet written.")
        return

    for name, df in [(f"av_fixed5_full_{args.tag}", full_df),
                     (f"av_fixed5_match_{args.tag}", match_df)]:
        d = OUT_BASE / name
        d.mkdir(parents=True, exist_ok=True)
        out = d / "merged.parquet"
        df.to_parquet(out, engine="pyarrow", index=False)
        print(f"wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
