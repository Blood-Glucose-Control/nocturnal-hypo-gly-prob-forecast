# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
CLI entry point for summarizing experiments.

Walks one or both experiment trees, writes summary CSVs, and prints the
best-run leaderboards to stdout.

Usage
-----
Run from the repository root::

    python scripts/analysis/summarize_experiments.py --type standard_forecasting
    python scripts/analysis/summarize_experiments.py --type nocturnal_forecasting
    python scripts/analysis/summarize_experiments.py --type all --metric mae
    python scripts/analysis/summarize_experiments.py --type all --root /abs/path/to/experiments

Module: scripts.analysis.summarize_experiments
Author: Blood-Glucose-Control
Created: 2026-02-22
"""

from __future__ import annotations

# Standard library imports
import argparse
import logging
from pathlib import Path

# Local imports
from src.experiments.base.experiment import VALID_METRICS
from src.experiments.nocturnal.summarize import NocturnalSummarizer
from src.experiments.standard_forecasting.summarize import StandardForecastingSummarizer

_SUMMARIZERS = {
    "standard_forecasting": StandardForecastingSummarizer,
    "nocturnal_forecasting": NocturnalSummarizer,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize experiment runs and print best-run leaderboards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=list(_SUMMARIZERS) + ["all"],
        default="all",
        dest="experiment_type",
        help="Which experiment tree to summarize.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(VALID_METRICS),
        default="rmse",
        help="Metric used to rank and select the best run (lower is better).",
    )
    parser.add_argument(
        "--root",
        default="experiments",
        help="Path to the top-level experiments/ directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _run_summarizer(name: str, cls, root: str, metric: str) -> None:
    """Run a single experiment summarizer and print results to stdout.

    Instantiates *cls* with *root* as the experiments directory, calls
    :meth:`best_runs` with the requested *metric*, and prints two leaderboard
    tables:

    - **Best run per (model × dataset)** — the single run with the lowest
      *metric* value for every ``(model, dataset)`` combination found.
    - **Global best run per model** — the single run with the lowest *metric*
      value for each model type across all datasets.

    CSV files (``summary.csv``, ``best_by_model_dataset.csv``,
    ``best_by_model.csv``) are written into ``<root>/<name>/`` as a side
    effect of :meth:`best_runs`.

    Parameters
    ----------
    name:
        Experiment type name, e.g. ``"standard_forecasting"``.  Used as the
        subdirectory name under *root* and in the printed header.
    cls:
        Concrete :class:`~src.experiments.base.experiment.ExperimentSummarizer`
        subclass to instantiate.
    root:
        Path to the top-level ``experiments/`` directory.
    metric:
        Metric column used for ranking (e.g. ``"rmse"``).  Must be one of
        :data:`~src.experiments.base.experiment.VALID_METRICS`.
    """
    print(f"\n{'='*70}")
    print(f"  Experiment type : {name}")
    print(f"  Metric          : {metric}")
    print(f"  Root            : {root}")
    print(f"{'='*70}")

    summarizer = cls(experiments_root=root)
    results = summarizer.best_runs(metric=metric)

    by_model_dataset = results["by_model_dataset"]
    by_model = results["by_model"]

    if by_model_dataset.empty:
        print("  [No completed runs found]\n")
        return

    print("\n--- Best run per (model × dataset) ---")
    print(
        by_model_dataset[
            [
                c
                for c in ["model", "dataset", "mode", "ctx_fh", metric, "run_id"]
                if c in by_model_dataset.columns
            ]
        ].to_string(index=False)
    )

    print("\n--- Global best run per model ---")
    print(
        by_model[
            [
                c
                for c in ["model", "dataset", "mode", "ctx_fh", metric, "run_id"]
                if c in by_model.columns
            ]
        ].to_string(index=False)
    )

    out_dir = Path(root) / name
    print(
        f"\n  CSV files written to {out_dir}/\n"
        f"    summary.csv\n"
        f"    best_by_model_dataset.csv\n"
        f"    best_by_model.csv\n"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
    )

    types_to_run = (
        list(_SUMMARIZERS.items())
        if args.experiment_type == "all"
        else [(args.experiment_type, _SUMMARIZERS[args.experiment_type])]
    )

    for name, cls in types_to_run:
        _run_summarizer(name, cls, args.root, args.metric)


if __name__ == "__main__":
    main()
