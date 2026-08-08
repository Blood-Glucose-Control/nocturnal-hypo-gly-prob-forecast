#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Count midnight episodes under different gap-fill tolerances per dataset.

For each dataset, builds episodes with three policies:
  * strict        – production default (max_bg_gap_steps=11 ≈ 55 min)
  * gap_60min     – up to 12 consecutive missing 5-min steps (= 1 hour)
  * theoretical   – an upper bound: count of calendar midnights in each
                    patient's recording window minus 2 warmup days minus 1
                    trailing day. Assumes "perfect data" (no gaps anywhere).

Outputs a small CSV at ``results/grand_summary/episode_count_sensitivity.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.versioning.dataset_registry import DatasetRegistry  # noqa: E402
from src.evaluation.episode_builders import build_midnight_episodes  # noqa: E402

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]
DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"


def _theoretical_per_patient(
    pdf: pd.DataFrame, *, context_length: int, forecast_length: int, interval_mins: int
) -> int:
    """Count calendar midnights inside [first+ctx, last-fh) for one patient."""
    if (
        pdf.empty
        or "datetime" not in pdf.columns
        and not isinstance(pdf.index, pd.DatetimeIndex)
    ):
        return 0
    if "datetime" in pdf.columns:
        idx = pd.to_datetime(pdf["datetime"]).sort_values()
    else:
        idx = pd.DatetimeIndex(pdf.index).sort_values()
    if len(idx) == 0:
        return 0
    dt = pd.Timedelta(minutes=interval_mins)
    earliest = idx.min() + context_length * dt
    latest = idx.max() - (forecast_length - 1) * dt
    first_midnight = earliest.normalize()
    if first_midnight < earliest:
        first_midnight += pd.Timedelta(days=1)
    last_midnight = latest.normalize()
    if last_midnight < first_midnight:
        return 0
    return int((last_midnight - first_midnight).days) + 1


def _build_count(
    df: pd.DataFrame,
    *,
    max_bg_gap_steps: int,
    context_length: int,
    forecast_length: int,
    target_col: str,
    interval_mins: int,
    patient_col: str,
) -> int:
    n = 0
    for _, pdf in df.groupby(patient_col):
        pdf = pdf.copy()
        if not isinstance(pdf.index, pd.DatetimeIndex):
            if "datetime" in pdf.columns:
                pdf["datetime"] = pd.to_datetime(pdf["datetime"])
                pdf = pdf.set_index("datetime").sort_index()
            else:
                continue
        episodes, _ = build_midnight_episodes(
            pdf,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=None,
            interval_mins=interval_mins,
            max_bg_gap_steps=max_bg_gap_steps,
        )
        for ep in episodes:
            tgt = np.asarray(ep["target_bg"])
            if len(tgt) >= forecast_length:
                n += 1
    return n


def _theoretical_count(
    df: pd.DataFrame,
    *,
    context_length: int,
    forecast_length: int,
    interval_mins: int,
    patient_col: str,
) -> int:
    n = 0
    for _, pdf in df.groupby(patient_col):
        n += _theoretical_per_patient(
            pdf,
            context_length=context_length,
            forecast_length=forecast_length,
            interval_mins=interval_mins,
        )
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--forecast-length", type=int, default=96)
    ap.add_argument("--interval-mins", type=int, default=5)
    ap.add_argument("--target-col", default="bg_mM")
    ap.add_argument("--patient-col", default="p_num")
    ap.add_argument(
        "--output",
        default="results/grand_summary/episode_count_sensitivity.csv",
    )
    args = ap.parse_args()

    registry = DatasetRegistry(holdout_config_dir=Path(args.config_dir))

    rows = []
    for ds in args.datasets:
        train_data, holdout_data = registry.load_dataset_with_split(
            ds, patient_col=args.patient_col
        )
        full = pd.concat([train_data, holdout_data], ignore_index=False)
        n_pat = int(full[args.patient_col].nunique())
        kw = dict(
            context_length=args.context_length,
            forecast_length=args.forecast_length,
            target_col=args.target_col,
            interval_mins=args.interval_mins,
            patient_col=args.patient_col,
        )
        n_strict = _build_count(full, max_bg_gap_steps=11, **kw)
        n_60 = _build_count(full, max_bg_gap_steps=12, **kw)
        n_theoretical = _theoretical_count(
            full,
            context_length=args.context_length,
            forecast_length=args.forecast_length,
            interval_mins=args.interval_mins,
            patient_col=args.patient_col,
        )
        rows.append(
            {
                "dataset": ds,
                "n_patients": n_pat,
                "strict_55min_gap_cap": n_strict,
                "gap_60min_cap": n_60,
                "theoretical_no_gaps": n_theoretical,
                "strict_yield_vs_theoretical": (
                    n_strict / n_theoretical if n_theoretical else float("nan")
                ),
                "gap60_yield_vs_theoretical": (
                    n_60 / n_theoretical if n_theoretical else float("nan")
                ),
            }
        )
        print(rows[-1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
