#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Build the three-way cohort variance/stats CSV.

For each dataset under a holdout ablation (default
``configs/data/holdout_10pct``), reports BG distribution stats for:

  * patient_holdout  – patients in YAML's ``patient_config.holdout_patients``
                       (entirely held out)
  * temporal_holdout – tail-end of the non-holdout patients
                       (the temporal split in the hybrid holdout)
  * temporal_train   – the rest (the actual training data)

Columns:
  ``cohort, dataset, n_patients, n_rows, n_episodes, n_episodes_with_hypo,
   hypo_rate, mean_bg, std_bg, mean_min_bg``

Episodes are built by :func:`build_midnight_episodes` with the default
context/forecast lengths used by the headline analysis (512 / 96), so
counts are directly comparable to ``dataset_holdout_characteristics.csv``.
``mean_min_bg`` is the average per-episode minimum BG across the forecast
window, matching how ``has_hypo`` is defined.

Output: ``<output_dir>/dataset_cohort_variance.csv``.

Example::

    python scripts/analysis/build_dataset_cohort_variance.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.grand_summary import DATASETS  # noqa: E402
from src.experiments.nocturnal.holdout_split_analysis import (  # noqa: E402
    build_three_way_cohort_stats,
)

DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"
DEFAULT_OUTPUT_DIR = "results/grand_summary"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--forecast-length", type=int, default=96)
    p.add_argument("--target-col", type=str, default="bg_mM")
    p.add_argument("--interval-mins", type=int, default=5)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ds in args.datasets:
        logging.info("Processing %s ...", ds)
        rows.extend(
            build_three_way_cohort_stats(
                args.config_dir,
                ds,
                context_length=args.context_length,
                forecast_length=args.forecast_length,
                target_col=args.target_col,
                interval_mins=args.interval_mins,
            )
        )

    df = pd.DataFrame(rows)
    out_path = out_dir / "dataset_cohort_variance.csv"
    df.to_csv(out_path, index=False)

    print()
    print("=" * 80)
    print(
        f"DATASET COHORT VARIANCE STATS "
        f"(config_dir={args.config_dir}, ctx={args.context_length}, fh={args.forecast_length})"
    )
    print("=" * 80)
    if df.empty:
        print("(no rows)")
    else:
        cols = [
            "dataset",
            "cohort",
            "n_patients",
            "n_episodes",
            "n_episodes_with_hypo",
            "hypo_rate",
            "mean_bg",
            "std_bg",
            "mean_min_bg",
        ]
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
