# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Export an Optuna study's trials to CSV/Parquet for offline review.

Flattens every completed trial's sampled parameters and recorded metrics
(RMSE/MARD/DTS at each horizon, covariate selection, GPU) into one table sorted
by the primary objective (RMSE@120). Use after (or during) a study to inspect
which configurations win.

Example::

    python -m src.workflows.sweeps.optuna_chronos2.export_results \\
        --storage sqlite:///results/optuna/chronos2_full.db \\
        --study-name chronos2_metabonet_full \\
        --out results/optuna/chronos2_full_trials
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import optuna
import pandas as pd


def _objective_columns(study: optuna.Study) -> list[str]:
    """Objective column name(s), derived from the study's metric name(s)."""
    names = list(getattr(study, "metric_names", None) or [])
    if not names:
        names = ["value"]
    return [f"objective_{m}" for m in names]


def study_to_frame(study: optuna.Study) -> pd.DataFrame:
    """Flatten a study's trials into a single DataFrame (single/multi-objective)."""
    obj_cols = _objective_columns(study)
    multi = len(obj_cols) > 1
    rows = []
    for t in study.trials:
        row: Dict[str, Any] = {"trial": t.number, "state": t.state.name}
        if multi:
            values = t.values if t.values is not None else [None] * len(obj_cols)
            for col, val in zip(obj_cols, values):
                row[col] = val
        else:
            row[obj_cols[0]] = t.value
        for key, value in t.params.items():
            row[f"param__{key}"] = value
        for key, value in t.user_attrs.items():
            row[f"attr__{key}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not multi and obj_cols[0] in frame.columns:
        frame = frame.sort_values(obj_cols[0], na_position="last")
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="Optuna storage URL.")
    parser.add_argument("--study-name", required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="Output path stem (writes <out>.csv and <out>.parquet).",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to print.")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    frame = study_to_frame(study)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(out.with_suffix(".parquet"), index=False)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        print(f"(parquet export skipped: {exc})")

    complete = frame[frame["state"] == "COMPLETE"]
    print(f"Trials: {len(frame)} total, {len(complete)} complete")
    cols = [
        c
        for c in [
            "trial",
            *_objective_columns(study),
            "attr__rmse_30",
            "attr__rmse_60",
            "attr__rmse_90",
            "attr__mard_120",
            "attr__dts_a_pct_120",
            "param__context_length",
            "param__fine_tune_batch_size",
            "param__fine_tune_steps",
        ]
        if c in frame.columns
    ]
    with pd.option_context("display.max_rows", args.top, "display.width", 160):
        print(complete[cols].head(args.top).to_string(index=False))
    print(f"\nWrote {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
