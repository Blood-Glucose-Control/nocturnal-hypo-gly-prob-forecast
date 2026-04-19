# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Three-tier storage for nocturnal evaluation results.

Splits the monolithic nocturnal_results.json into three purpose-built tiers:

  Tier 1 — ``results_summary.json``: Overall scalar metrics + run metadata.
      Used by the summarizer for leaderboard CSV and quick comparisons.

  Tier 2 — ``episodes.parquet``: One row per episode with scalar metrics only
      (no arrays). Supports per-patient breakdowns, box plots, distributional
      analysis. ~1 MB per 5000 episodes.

  Tier 3 — ``forecasts.npz``: Compressed raw quantile forecast arrays.
      Written only for probabilistic runs. Enables post-hoc PIT histograms,
      reliability diagrams, and re-calibration without re-running the model.
      ~6 MB per run.

Usage::

    from src.evaluation.storage import write_nocturnal_results

    results = evaluate_nocturnal_forecasting(model, data, ..., probabilistic=True)
    metadata = {"model": "chronos2", "dataset": "brown_2019", ...}
    written = write_nocturnal_results(results, output_path, metadata)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics.shape import DILATE_COLUMNS

logger = logging.getLogger(__name__)

# Columns to strip from per-episode dicts before writing Tier 2.
# These are large arrays that belong in Tier 3 (or are reconstructable).
_ARRAY_COLUMNS = {"pred", "target_bg", "context_bg"}


def write_nocturnal_results(
    results: Dict[str, Any],
    output_path: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    """Write evaluation results as 3-tier files.

    Args:
        results: Return dict from ``evaluate_nocturnal_forecasting()``.
        output_path: Run directory (must already exist).
        metadata: Run-level fields (model, dataset, config, timestamp, etc.)
            merged into Tier 1.

    Returns:
        Dict mapping tier name to the Path written.
    """
    output_path = Path(output_path)
    written: Dict[str, Path] = {}

    written["tier1"] = _write_tier1(results, metadata, output_path)
    written["tier2"] = _write_tier2(results, output_path)
    written["tier3"] = _write_tier3(results, output_path)

    return written


def _write_tier1(
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Write ``results_summary.json`` - overall scalars + metadata.

    Includes per_patient aggregates (small) but excludes per_episode arrays.
    """
    summary: Dict[str, Any] = {**metadata}

    # Overall scalar metrics
    for key in (
        "overall_rmse",
        "mean_discontinuity",
        "total_episodes",
        "overall_wql",
        "overall_brier",
        "overall_mace",
        "overall_coverage_50",
        "overall_coverage_80",
        "overall_sharpness_50",
        "overall_sharpness_80",
        "quantile_levels",
    ) + tuple(f"overall_{col}" for col in DILATE_COLUMNS):
        if key in results:
            summary[key] = results[key]

    # Per-patient aggregates are small — include for quick analysis
    if "per_patient" in results:
        summary["per_patient"] = results["per_patient"]

    path = output_path / "results_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Tier 1 (summary): %s", path)
    return path


def _write_tier2(
    results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Write ``episodes.parquet`` - per-episode scalar metrics, no arrays."""
    per_episode: List[Dict[str, Any]] = results.get("per_episode", [])
    # Strip array columns
    rows = [
        {k: v for k, v in ep.items() if k not in _ARRAY_COLUMNS} for ep in per_episode
    ]
    df = pd.DataFrame(rows)

    path = output_path / "episodes.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Tier 2 (episodes): %s  (%d rows)", path, len(df))
    return path


def _write_tier3(
    results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Write ``forecasts.npz`` - compressed raw forecast arrays.

    Always written. Contents:
        predictions:        (n_episodes, forecast_length) — point forecasts
                            (median quantile when probabilistic)
        actuals:            (n_episodes, forecast_length)
        episode_ids:        (n_episodes,) fixed-width unicode
        quantile_forecasts: (n_episodes, n_quantiles, forecast_length)
                            Empty array when run was non-probabilistic.
        quantile_levels:    (n_quantiles,) float
                            Empty array when run was non-probabilistic.
    """
    predictions = results["_predictions"]  # (n_eps, fh)
    actuals = results["_actuals_array"]  # (n_eps, fh)
    # dtype=str → fixed-width unicode, avoids pickling (allow_pickle=False-safe)
    episode_ids = np.array(results["_episode_ids"], dtype=str)
    q_forecasts = results.get("_q_forecasts", np.empty((0, 0, 0), dtype=np.float64))
    quantile_levels = np.array(results.get("quantile_levels", []), dtype=np.float64)

    path = output_path / "forecasts.npz"
    np.savez_compressed(
        path,
        predictions=predictions,
        actuals=actuals,
        episode_ids=episode_ids,
        quantile_forecasts=q_forecasts,
        quantile_levels=quantile_levels,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Tier 3 (forecasts): %s  (%.1f MB)", path, size_mb)
    return path
