"""ALTERNATIVE Fig 3 (2026-09-22): a single LL panel with the usual
truncated axis. Same data, series, encodings and caption machinery as
plot_thesis_fig3_transfer.py; NSS and AUC go to the table.
Output: fig3_transfer_ll[_test].*; the other variants are not overwritten.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_thesis_fig3_transfer as base

if __name__ == "__main__":
    base.main(
        metrics=[("ll_img", "LL (bits/fixation) \u2191")],
        name="fig3_transfer_ll",
        ours_label="Our model",
        legend_model_columns=True,
        filled_markers=True,
        marker_size=4.0,
        extra_caption=(
            " Horizontal lines are reference conditions (no pretraining; "
            "DeepGaze III's SALICON pretraining) and are not associated "
            "with a point on the x axis."),
    )
