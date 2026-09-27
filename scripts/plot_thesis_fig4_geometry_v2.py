"""Thesis Fig 4 (v2, 2026-09-17): scanpath statistics of the three gaze
sources, laid out like DeepGaze III's Fig. 3 (Kümmerer, Bethge & Wallis
2022, J. Vision 22(5):7): linear plots throughout, one panel per statistic.

  A  saccade amplitude          image-diagonal fraction (no dva axis: the
                                synthetic sources have no viewing geometry;
                                for MIT1003 alone 1.0 diagonal ~ 35 dva)
  B  distance from image centre image-diagonal fraction (our panel, kept
                                by decision; not in the DG3 set)
  C  saccade direction          linear axis left / down / right / up / left
                                (DG3 Fig. 3c), isotropic coordinates
  D  angle between saccades     signed, -180..180 (DG3 Fig. 3e); 0 = same
                                direction, +-180 = return saccade
The amplitude autocorrelation (DG3 Fig. 3g) is a scalar in
scanpath_geometry.csv and goes in the text, not in the figure.

Reads only docs/report_tables/scanpath_geometry_hists.csv. Human =
MIT1003 train+validation; first fixation dropped from every source.
The script accepts both the old histogram layout (15-degree direction
bins, unsigned 0..180 angle) and the new one (5-degree bins, signed
angle) and labels panel C accordingly.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_fig_style import COLORS, RT, apply_style, panel_letter, save

SOURCES = [("human", "human (MIT1003)", COLORS["ref"]),
           ("av", "active vision", COLORS["ours"]),
           ("scandiff", "ScanDiff", COLORS["scandiff"])]


def load_hists():
    data = defaultdict(lambda: defaultdict(list))
    with (RT / "scanpath_geometry_hists.csv").open() as fh:
        for row in csv.DictReader(fh):
            data[row["stat"]][row["source"]].append(
                (float(row["bin_left"]), float(row["bin_right"]), float(row["density"])))
    for stat in data:
        for src in data[stat]:
            data[stat][src].sort()
    return data


def line_panel(ax, rows_by_source, xlabel, transform=None, closed=False):
    for source, label, color in SOURCES:
        rows = rows_by_source[source]
        pts = [((l + r) / 2, d) for l, r, d in rows]
        if transform is not None:
            pts = sorted((transform(x), d) for x, d in pts)
        if closed and pts:   # periodic axis: repeat the first bin at the far end
            x0, d0 = pts[0]
            pts = pts + [(x0 + 360.0, d0)]
        ax.plot([x for x, _ in pts], [d for _, d in pts], color=color, lw=1.6,
                label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)


def main():
    apply_style()
    data = load_hists()
    signed = min(l for l, _, _ in data["angle"]["human"]) < 0

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    line_panel(ax_a, data["sacc"], "Saccade amplitude (diagonal fraction)")
    panel_letter(ax_a, "A")

    # B distance from centre
    line_panel(ax_b, data["r"], "Distance from center (diagonal fraction)")
    panel_letter(ax_b, "B")

    # C direction: 0 = right, 90 = up in the table -> paper axis left..left
    line_panel(ax_c, data["direction"], "Saccade direction",
               transform=lambda a: ((a + 180.0) % 360.0) - 180.0, closed=True)
    ax_c.set_xticks([-180, -90, 0, 90, 180])
    ax_c.set_xticklabels(["left", "down", "right", "up", "left"])
    ax_c.set_xlim(-180, 180)
    ax_c.axhline(1.0 / 360.0, color="#777777", lw=0.9, ls="--", zorder=1)
    panel_letter(ax_c, "C")

    # D angle between consecutive saccades
    if signed:
        line_panel(ax_d, data["angle"], "Angle between saccades (°)", closed=True)
        ax_d.set_xticks([-180, -90, 0, 90, 180])
        ax_d.set_xticklabels(["±180", "−90", "0", "90", "±180"])
        ax_d.set_xlim(-180, 180)
        ax_d.axhline(1.0 / 360.0, color="#777777", lw=0.9, ls="--", zorder=1)
    else:
        line_panel(ax_d, data["angle"], "Angle between saccades (°, unsigned)")
        ax_d.set_xticks([0, 45, 90, 135, 180])
        ax_d.set_xlim(0, 180)
        ax_d.axhline(1.0 / 180.0, color="#777777", lw=0.9, ls="--", zorder=1)
    panel_letter(ax_d, "D")

    handles = [Line2D([], [], color=color, lw=1.6, label=label)
               for _, label, color in SOURCES]
    handles.append(Line2D([], [], color="#777777", lw=0.9, ls="--",
                          label="uniform (C, D)"))
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.8, handlelength=2.4,
               fontsize=7.5)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.09, top=0.90,
                        hspace=0.50, wspace=0.38)

    dir_bins = len(data["direction"]["human"])
    angle_txt = ("(D) Signed angle between consecutive saccades (0° = same "
                 "direction, ±180° = return saccade, positive = counter-clockwise turn)."
                 if signed else
                 "(D) Unsigned angle between consecutive saccades (0° = same "
                 "direction, 180° = return saccade).")
    save(fig, "fig4_geometry_v2", caption=f"""
Fig 4 — Scanpath statistics of the three gaze sources (panels C and D
follow the conventions of DeepGaze III's Fig. 3): human scanpaths (MIT1003, training and validation
images), the active-vision pretraining corpus and the ScanDiff-generated
corpus (both on the same 100,000 ImageNet images). (A) Saccade amplitude as
a fraction of the image diagonal. (B) Distance of fixations from the image centre as a
fraction of the image diagonal. (C) Saccade direction in isotropic
coordinates ({360 // dir_bins}° bins), so the aspect ratio of the human
stimuli does not compress horizontal saccades. {angle_txt} The
first fixation of every scanpath is excluded from all sources (forced
central start in the active-vision protocol and the MIT1003 fixation
cross). Each curve is a probability density and integrates to one; the dashed
line in C and D marks the uniform distribution over angles.""")


if __name__ == "__main__":
    main()
