#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Build the per-dataset holdout-characteristics CSV.

For every dataset in a given holdout ablation (default
``configs/data/holdout_10pct``), counts:

  * how many patients fall in the patient- vs temporal-holdout split,
  * how many midnight episodes were evaluated for each split, and
  * how many of those episodes contained at least one true BG reading
    below 3.9 mmol/L (the standard hypoglycemia threshold).

The episode set is model-invariant for a given holdout config (it depends
only on the data, not the forecaster), so we pick *one arbitrary* run per
dataset that used the requested ``config_dir`` and read its
``episodes.parquet`` + ``forecasts.npz``.

Output: ``<output_dir>/dataset_holdout_characteristics.csv`` with columns
``[ablation, dataset, split_type, n_patients, n_episodes,
n_episodes_with_hypo, hypo_rate, source_run]``.

Example::

    python scripts/analysis/build_dataset_holdout_characteristics.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.grand_summary import DATASETS  # noqa: E402
from src.experiments.nocturnal.holdout_split_analysis import (  # noqa: E402
    SPLIT_PATIENT,
    SPLIT_TEMPORAL,
    load_run_episode_classification,
)

DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"
DEFAULT_EXPERIMENTS_ROOTS = (
    "experiments/nocturnal_forecasting",
    "experiments/nocturnal_forecasting_ctx_ablation",
)
DEFAULT_OUTPUT_DIR = "results/grand_summary"

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    p.add_argument(
        "--experiments-root",
        action="append",
        default=None,
        help=(
            "Root directory holding nocturnal-forecasting runs (repeatable). "
            f"Default: {' + '.join(DEFAULT_EXPERIMENTS_ROOTS)}"
        ),
    )
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def find_one_run_per_dataset(
    experiments_roots: list[Path],
    config_dir: str,
    datasets: list[str],
) -> dict[str, Path]:
    """Walk experiment-config jsons; return the first matching run per dataset.

    A run matches if its ``cli_args.config_dir == config_dir`` and its
    ``cli_args.dataset`` is one of ``datasets`` and its run dir contains
    both ``episodes.parquet`` and ``forecasts.npz``.
    """
    found: dict[str, Path] = {}
    needed = set(datasets)
    for root in experiments_roots:
        if not root.exists():
            log.warning("Experiments root %s does not exist; skipping.", root)
            continue
        for cfg_path in root.rglob("experiment_config.json"):
            if not needed:
                return found
            try:
                with cfg_path.open() as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            cli = cfg.get("cli_args", {}) or {}
            if str(cli.get("config_dir", "")) != config_dir:
                continue
            ds = str(cli.get("dataset", ""))
            if ds not in needed:
                continue
            run_dir = cfg_path.parent
            if not (run_dir / "episodes.parquet").exists():
                continue
            if not (run_dir / "forecasts.npz").exists():
                continue
            found[ds] = run_dir
            needed.discard(ds)
    return found


def summarize_run(run_dir: Path, config_dir: str, dataset: str) -> list[dict]:
    df = load_run_episode_classification(
        run_dir, config_dir=config_dir, dataset=dataset
    )
    rows: list[dict] = []
    for split in (SPLIT_PATIENT, SPLIT_TEMPORAL):
        sub = df[df["split_type"] == split]
        n_eps = int(len(sub))
        n_pts = int(sub["patient_id"].nunique())
        n_hypo = int(sub["has_hypo"].sum())
        rows.append(
            {
                "ablation": Path(config_dir).name,
                "dataset": dataset,
                "split_type": split,
                "n_patients": n_pts,
                "n_episodes": n_eps,
                "n_episodes_with_hypo": n_hypo,
                "hypo_rate": (n_hypo / n_eps) if n_eps else float("nan"),
                "source_run": str(run_dir),
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    experiments_roots = [
        Path(r) for r in (args.experiments_root or DEFAULT_EXPERIMENTS_ROOTS)
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_one_run_per_dataset(experiments_roots, args.config_dir, args.datasets)
    missing = [ds for ds in args.datasets if ds not in runs]
    if missing:
        log.error(
            "No matching run found for datasets %s under config_dir=%s",
            missing,
            args.config_dir,
        )

    rows: list[dict] = []
    for ds in args.datasets:
        if ds not in runs:
            continue
        log.info("Using %s for %s", runs[ds], ds)
        rows.extend(summarize_run(runs[ds], args.config_dir, ds))

    df = pd.DataFrame(rows)
    out_path = out_dir / "dataset_holdout_characteristics.csv"
    df.to_csv(out_path, index=False)

    print()
    print("=" * 80)
    print(f"DATASET HOLDOUT CHARACTERISTICS (config_dir={args.config_dir})")
    print("=" * 80)
    if df.empty:
        print("(no rows)")
    else:
        print(
            df.drop(columns=["source_run"]).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"
            )
        )
    print(f"\nWrote {out_path.resolve()}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
