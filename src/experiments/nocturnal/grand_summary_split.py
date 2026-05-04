# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Best-run breakdown across the patient/temporal holdout split.

This module reuses the leaderboard plumbing in
``src.experiments.nocturnal.grand_summary`` to identify the best
(model × dataset × cov_bucket) run for a given holdout ablation, then
re-aggregates the per-episode metrics in each best run separately for the
patient-holdout vs temporal-holdout subsets.

It does **not** mutate ``summary.csv`` or modify ``grand_summary.py``.

Aggregation rules (matching ``src/evaluation/nocturnal.py``):
  * ``rmse``                – sqrt(mean(squared_error)) over the split's
                              (episode, step) pairs, recomputed from the
                              raw arrays in ``forecasts.npz``.
  * ``wql``, ``brier_3_9``, ``discontinuity``, ``coverage_*``,
    ``sharpness_*``, ``dilate_*``, ``shape_*``, ``temporal_*`` –
                              nanmean of the per-episode values in the split.
  * ``n_episodes``           – row count.
  * ``n_episodes_with_hypo`` – count of episodes whose actual BG dipped below
                              ``HYPO_THRESHOLD_MMOL`` at any forecast step.
  * ``n_patients``           – distinct patient_id count.

``mace`` is omitted because it requires the per-step quantile arrays
across all episodes in the split; it is only available at the overall level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.experiments.nocturnal.grand_summary import (
    DATASETS,
    MODEL_PROPERTIES,
    attach_covariate_bucket,
    dedupe,
    load_summary,
    pick_best,
)
from src.experiments.nocturnal.holdout_split_analysis import (
    HYPO_THRESHOLD_MMOL,
    SPLIT_PATIENT,
    SPLIT_TEMPORAL,
    load_run_episode_classification,
)

log = logging.getLogger(__name__)

# Per-episode metric columns to average within a split.
_MEAN_METRICS_REQUIRED: tuple[str, ...] = ("discontinuity",)
_MEAN_METRICS_OPTIONAL: tuple[str, ...] = (
    "wql",
    "brier",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "coverage_95",
    "sharpness_50",
    "sharpness_80",
    "sharpness_90",
    "sharpness_95",
    "dilate_g0001",
    "shape_g0001",
    "temporal_g0001",
    "dilate_g001",
    "shape_g001",
    "temporal_g001",
    "dilate_g01",
    "shape_g01",
    "temporal_g01",
)

_SPLITS: tuple[str, ...] = (SPLIT_PATIENT, SPLIT_TEMPORAL)


# ---------------------------------------------------------------------------
# Filtering by config_dir
# ---------------------------------------------------------------------------


def _read_config_dir(run_path: str) -> str | None:
    cfg_path = Path(run_path) / "experiment_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    cli = cfg.get("cli_args", {}) or {}
    cd = cli.get("config_dir")
    return str(cd) if cd else None


def attach_config_dir(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``config_dir`` column derived from each run's experiment config.

    Cached per ``run_path`` to avoid re-reading the same file.
    """
    cache: dict[str, str | None] = {}
    cds: list[str | None] = []
    for run_path in df["run_path"].astype(str):
        if run_path not in cache:
            cache[run_path] = _read_config_dir(run_path)
        cds.append(cache[run_path])
    out = df.copy()
    out["config_dir"] = cds
    return out


def filter_to_config_dir(df: pd.DataFrame, config_dir: str) -> pd.DataFrame:
    """Keep only rows whose run was launched with ``--config-dir == config_dir``."""
    if "config_dir" not in df.columns:
        df = attach_config_dir(df)
    return df[df["config_dir"] == config_dir].copy()


# ---------------------------------------------------------------------------
# Per-run split aggregation
# ---------------------------------------------------------------------------


def _split_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    if predictions.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def aggregate_run_by_split(
    run_path: str | Path,
    *,
    model: str,
    dataset: str,
    cov_bucket: str,
    config_dir: str,
) -> list[dict]:
    """Return one row per ``split_type`` for the given run."""
    df = load_run_episode_classification(
        run_path, config_dir=config_dir, dataset=dataset
    )
    rows: list[dict] = []
    for split in _SPLITS:
        sub = df[df["split_type"] == split]
        if sub.empty:
            row: dict = {
                "model": model,
                "dataset": dataset,
                "cov_bucket": cov_bucket,
                "split_type": split,
                "n_patients": 0,
                "n_episodes": 0,
                "n_episodes_with_hypo": 0,
                "hypo_rate": float("nan"),
                "rmse": float("nan"),
                "run_path": str(run_path),
            }
            for col in _MEAN_METRICS_REQUIRED + _MEAN_METRICS_OPTIONAL:
                row[col if col != "brier" else "brier_3_9"] = float("nan")
            rows.append(row)
            continue

        preds = np.stack(sub["predictions"].to_numpy())
        actuals = np.stack(sub["actuals"].to_numpy())
        n_eps = int(len(sub))
        n_hypo = int(sub["has_hypo"].sum())
        row = {
            "model": model,
            "dataset": dataset,
            "cov_bucket": cov_bucket,
            "split_type": split,
            "n_patients": int(sub["patient_id"].nunique()),
            "n_episodes": n_eps,
            "n_episodes_with_hypo": n_hypo,
            "hypo_rate": (n_hypo / n_eps) if n_eps else float("nan"),
            "rmse": _split_rmse(preds, actuals),
            "run_path": str(run_path),
        }
        for col in _MEAN_METRICS_REQUIRED:
            row[col] = float(sub[col].mean(skipna=True)) if col in sub else float("nan")
        for col in _MEAN_METRICS_OPTIONAL:
            out_col = "brier_3_9" if col == "brier" else col
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[out_col] = (
                    float(vals.mean(skipna=True))
                    if vals.notna().any()
                    else float("nan")
                )
            else:
                row[out_col] = float("nan")
        rows.append(row)
    return rows


def _model_class(model: str) -> str:
    props = MODEL_PROPERTIES.get(model)
    if props:
        return str(props["class"])
    parent = model.split("/", 1)[0]
    parent_props = MODEL_PROPERTIES.get(parent)
    if parent_props:
        return str(parent_props["class"])
    return "deep_learning"


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_best_split_breakdown(
    summary_paths: Sequence[str | Path],
    config_dir: str,
    *,
    datasets: Sequence[str] = DATASETS,
    ctx_filter: int | None = None,
    forecast_filter: int | None = 96,
) -> pd.DataFrame:
    """Build the long-form best-runs split breakdown for one holdout ablation.

    Parameters mirror :func:`grand_summary.build_grand_summary`.

    Returns a DataFrame with one row per
    ``(model, dataset, cov_bucket, split_type)``.
    """
    raw = load_summary(summary_paths)
    if forecast_filter is not None and "forecast_length" in raw.columns:
        raw = raw[raw["forecast_length"] == forecast_filter]
    if ctx_filter is not None and "context_length" in raw.columns:
        raw = raw[raw["context_length"] == ctx_filter]
    raw = raw[raw["model"].isin(MODEL_PROPERTIES)]
    raw = raw[raw["dataset"].isin(list(datasets))]
    raw = attach_config_dir(raw)
    raw = raw[raw["config_dir"] == config_dir]
    if raw.empty:
        log.warning(
            "No runs match config_dir=%s after filtering; returning empty frame.",
            config_dir,
        )
        return pd.DataFrame()
    deduped = dedupe(raw)
    bucketed = attach_covariate_bucket(deduped)

    # Build a ranked list of all candidates per group (same sort as pick_best).
    # We walk down the ranking until we find a run that has episodes.parquet,
    # so that old runs missing the file don't silently drop the whole group.
    # GROUP_COLS = ["model", "dataset", "cov_bucket"]
    ranked = (
        pick_best.__wrapped__(bucketed)
        if hasattr(pick_best, "__wrapped__")
        else bucketed.copy()
    )
    ranked["rmse"] = pd.to_numeric(ranked["rmse"], errors="coerce")
    ranked["wql"] = pd.to_numeric(ranked["wql"], errors="coerce")
    ranked["timestamp"] = pd.to_datetime(ranked["timestamp"], errors="coerce")
    ranked = ranked.dropna(subset=["rmse"]).sort_values(
        ["rmse", "wql", "timestamp"], ascending=[True, True, False], na_position="last"
    )
    candidates: dict[tuple, list] = {}
    for _, r in ranked.iterrows():
        key = (str(r["model"]), str(r["dataset"]), str(r["cov_bucket"]))
        candidates.setdefault(key, []).append(r)

    rows: list[dict] = []
    for key, cands in candidates.items():
        model_name, dataset_name, cov_bucket_name = key
        split_rows = None
        for r in cands:
            run_path = str(r["run_path"])
            try:
                split_rows = aggregate_run_by_split(
                    run_path,
                    model=model_name,
                    dataset=dataset_name,
                    cov_bucket=cov_bucket_name,
                    config_dir=config_dir,
                )
                break  # success — use this run
            except (FileNotFoundError, ValueError) as exc:
                log.warning(
                    "Skipping %s (falling back to next-best): %s", run_path, exc
                )
        if split_rows is None:
            log.warning(
                "No usable run found for (%s, %s, %s) — all candidates missing episodes.parquet",
                model_name,
                dataset_name,
                cov_bucket_name,
            )
            continue
        # r is the last successfully used candidate row
        for sr in split_rows:
            sr["model_class"] = _model_class(sr["model"])
            sr["context_length"] = r.get("context_length")
            sr["forecast_length"] = r.get("forecast_length")
            sr["overall_rmse"] = (
                float(r.get("rmse")) if pd.notna(r.get("rmse")) else float("nan")
            )
            sr["overall_wql"] = (
                float(r.get("wql")) if pd.notna(r.get("wql")) else float("nan")
            )
            rows.append(sr)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["model_class", "model", "dataset", "cov_bucket", "split_type"]
        ).reset_index(drop=True)
        # Lead with the identifier/grouping columns so downstream sorting
        # and pivoting is straightforward.
        lead = [
            "model_class",
            "model",
            "dataset",
            "cov_bucket",
            "split_type",
            "n_patients",
            "n_episodes",
            "n_episodes_with_hypo",
            "hypo_rate",
            "rmse",
        ]
        rest = [c for c in out.columns if c not in lead]
        out = out[[c for c in lead if c in out.columns] + rest]
    return out


__all__ = [
    "HYPO_THRESHOLD_MMOL",
    "aggregate_run_by_split",
    "attach_config_dir",
    "build_best_split_breakdown",
    "filter_to_config_dir",
]
