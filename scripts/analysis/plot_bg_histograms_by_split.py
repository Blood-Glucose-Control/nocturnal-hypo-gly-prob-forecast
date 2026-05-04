#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Plot per-dataset BG histograms for train, temporal-holdout and patient-holdout splits.

Produces a 3-row x 4-column grid (rows=splits, cols=datasets) using bin edges
np.arange(0, 22.1, 0.1) on the ``bg_mM`` column.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.versioning.dataset_registry import DatasetRegistry  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]
SPLITS = ["train", "temporal_holdout", "patient_holdout"]
SPLIT_LABELS = {
    "train": "Train",
    "temporal_holdout": "Temporal",
    "patient_holdout": "Patient",
}
DATASET_LABELS = {
    "aleppo_2017": "REPLACE-BG",
    "brown_2019": "DCLP3",
    "lynch_2022": "IOBP2",
    "tamborlane_2008": "Tamborlane",
}
DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"
DEFAULT_OUTPUT = "results/grand_summary/bg_histograms_by_split.png"


def _split_three_ways(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_patients: set[str],
    patient_col: str,
) -> dict[str, pd.DataFrame]:
    """Decompose the registry's (train, holdout) into the three semantic cohorts."""
    holdout_df = holdout_df.copy()
    holdout_df[patient_col] = holdout_df[patient_col].astype(str)
    is_pat = holdout_df[patient_col].isin(holdout_patients)
    return {
        "train": train_df,
        "temporal_holdout": holdout_df.loc[~is_pat],
        "patient_holdout": holdout_df.loc[is_pat],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--target-col", default="bg_mM")
    p.add_argument("--patient-col", default="p_num")
    p.add_argument("--bin-min", type=float, default=2.0)
    p.add_argument("--bin-max", type=float, default=22.0)
    p.add_argument("--bin-width", type=float, default=0.5)
    p.add_argument("--hypo-threshold", type=float, default=3.9)
    p.add_argument("--hyper-threshold", type=float, default=10.0)
    p.add_argument(
        "--fig-width",
        type=float,
        default=6.75,
        help="Figure width in inches (NeurIPS column width).",
    )
    p.add_argument("--fig-height", type=float, default=4.25)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )

    bins = np.arange(args.bin_min, args.bin_max + args.bin_width, args.bin_width)
    registry = DatasetRegistry(holdout_config_dir=Path(args.config_dir))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )

    n_rows, n_cols = len(SPLITS), len(args.datasets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(args.fig_width, args.fig_height),
        sharex=True,
        sharey="row",
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    thousands_fmt = mticker.FuncFormatter(
        lambda x, _pos: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"
    )

    for c, ds in enumerate(args.datasets):
        logger.info("Loading %s", ds)
        train_df, holdout_df = registry.load_dataset_with_split(
            ds, patient_col=args.patient_col
        )
        config = registry.get_holdout_config(ds)
        holdout_patients = set(
            str(x)
            for x in (
                config.patient_config.holdout_patients
                if config and config.patient_config
                else []
            )
        )
        cohorts = _split_three_ways(
            train_df, holdout_df, holdout_patients, args.patient_col
        )

        for r, split_name in enumerate(SPLITS):
            ax = axes[r, c]
            df = cohorts[split_name]
            vals = df[args.target_col].to_numpy()
            vals = vals[np.isfinite(vals)]
            n_pat = df[args.patient_col].astype(str).nunique() if len(df) else 0
            counts, edges = np.histogram(vals, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            colors = np.where(
                centers < args.hypo_threshold,
                "tab:red",
                np.where(centers >= args.hyper_threshold, "tab:orange", "steelblue"),
            )
            ax.bar(
                centers,
                counts,
                width=args.bin_width,
                color=colors,
                edgecolor="none",
                align="center",
            )
            ax.set_xlim(args.bin_min, args.bin_max)
            ax.set_xticks(
                np.arange(args.bin_min, args.bin_max + 1e-9, 2.0), minor=False
            )
            ax.set_xticks(np.arange(args.bin_min, args.bin_max + 1e-9, 1.0), minor=True)
            ax.tick_params(axis="x", labelsize=6, pad=1.5)
            ax.tick_params(axis="x", which="minor", length=2)
            ax.tick_params(axis="x", which="major", length=3)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(0)
            ax.yaxis.set_major_formatter(thousands_fmt)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if r == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds))
            if c == 0:
                ax.set_ylabel(SPLIT_LABELS[split_name])
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel("BG (mmol/L)")
            ax.text(
                0.97,
                0.93,
                f"CGM Count={len(vals)/1000000:.1f}M\nPatients={n_pat}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4, alpha=0.85
                ),
            )

    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info("Wrote %s", out_path.resolve())


if __name__ == "__main__":
    main()
