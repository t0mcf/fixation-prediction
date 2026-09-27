"""Shared style for the thesis figures (scripts/plot_thesis_fig*.py).

Design decisions, made once here so the four figures cannot drift:

* Entity colors are FIXED across every figure (color follows the entity,
  never the panel): ours=blue, DeepGaze III=orange, ScanDiff=green; the
  Okabe–Ito CVD-safe triple, validated under deuteranomaly/protanomaly/
  tritanomaly simulation (CAM02-UCS ΔE ≥ 12 for all series pairs,
  2026-09-02). Pending/placeholder material is mid-grey AND dashed
  (color + linestyle, never color alone). Reference lines (ceiling,
  uniform, human) are near-black and unmarked.
* Marker shapes double as identity (circle/square/diamond) so series
  survive greyscale printing.
* Repetitions (decided 2026-09-17, one rule for every figure): the marker
  is the MEAN over runs, a thin capless line spans the MIN-MAX range;
  never caps (reads as a confidence interval), never one point per
  seed. A single run gets no line; the caption says so once.
* One y-axis per panel, light horizontal grid only, no top/right spines.
* Sizing targets a 160 mm text width (A4 thesis, \textwidth); base font
  9 pt so labels land near the caption size when included at full width.
* Output: vector PDF (the citable artifact) + 200 dpi PNG preview.

Figures read only docs/report_tables/*.csv (transitional exceptions are
declared in the individual script headers).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RT = REPO / "docs/report_tables"
OUT = REPO / "docs/figures_thesis"

COLORS = {
    "ours": "#0072B2",
    "dg3": "#E69F00",
    "scandiff": "#009E73",
    "pending": "#999999",
    "ref": "#222222",
}
MARKERS = {"ours": "o", "dg3": "s", "scandiff": "D"}
LABELS = {"ours": "ours", "dg3": "DeepGaze III", "scandiff": "ScanDiff"}

TEXTWIDTH_IN = 6.3  # 160 mm

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType, keeps text editable/searchable
}


SPLIT = "validation"   # set by apply_style(); "test" reads *_test.csv, writes *_test.*


def apply_style(split: str | None = None):
    """Set rcParams; parse --split from argv if not given explicitly.
    'test' (final evaluation only) switches every table read to its *_test.csv twin
    and suffixes the figure files, so validation figures are never overwritten."""
    global SPLIT
    if split is None:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--split", choices=["validation", "test"], default="validation")
        split = ap.parse_args().split
    SPLIT = split
    mpl.rcParams.update(RC)
    OUT.mkdir(parents=True, exist_ok=True)


def table(name: str, split_dependent: bool = True) -> Path:
    """Path of a report table for the active split. Split-independent tables
    (geometry, training-log peaks, ceiling) are passed with
    split_dependent=False and never get the suffix."""
    if SPLIT == "test" and split_dependent:
        return RT / name.replace(".csv", "_test.csv")
    return RT / name


# min-max range over repetitions: thin, capless (see module docstring)
RANGE_KW = dict(capsize=0, elinewidth=0.8, barsabove=False)

# Small horizontal dodge per entity on the log image axis, so range lines of
# different models at the same rung never run through each other's markers
# (2026-09-17). Same factors in every ladder figure: ours left, DG3 centre,
# ScanDiff right.
DODGE = {"ours": 0.94, "dg3": 1.0, "scandiff": 1.06}


def dodge_x(entity: str, xs):
    """Multiply rung positions by the entity's dodge factor (log axis)."""
    return [x * DODGE[entity] for x in xs]


def series_kw(entity: str, filled: bool = False) -> dict:
    """Line+marker kwargs for one of the three model entities.
    Hollow markers by default (2026-09-17): the white face hides the
    connecting line and the range line inside the marker, so only the
    part of the range that exceeds the marker is visible."""
    c = COLORS[entity]
    return dict(
        color=c,
        marker=MARKERS[entity],
        markerfacecolor=c if filled else "white",
        markeredgecolor=c,
        markeredgewidth=1.3,
        label=LABELS[entity],
        zorder=3,
    )


def scale_xaxis(ax):
    """Log x-axis in training images. Ticks are the ladder rungs: labeled
    at 1k/10k/100k/800k, unlabeled minor ticks at 50k/200k/400k."""
    import matplotlib.ticker as mticker
    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4, 1e5, 8e5])
    ax.set_xticklabels(["1k", "10k", "100k", "800k"])
    ax.xaxis.set_minor_locator(mticker.FixedLocator([5e4, 2e5, 4e5]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.5)
    ax.set_xlabel("synthetic training images")


def panel_letter(ax, letter: str):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=8)


def save(fig, name: str, caption: str | None = None):
    if SPLIT == "test":
        name = f"{name}_test"
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    if caption is not None:
        (OUT / f"{name}_caption.txt").write_text(caption.strip() + "\n")
    print(f"wrote {pdf.name}, {png.name}" + (", caption" if caption else ""))
