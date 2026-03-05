# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Summarizer for standard (sliding-window) forecasting experiments.

Scans all run directories under ``experiments/standard_forecasting/`` and
builds a tidy CSV with one row per run.  Extends
:class:`~src.experiments.base.experiment.ExperimentSummarizer` with logic to
parse ``results.json`` files as produced by
``scripts/experiments/sliding_window_eval.py``.

Module: experiments.standard_forecasting.summarize
Author: Blood-Glucose-Control
Created: 2026-02-22
"""

from __future__ import annotations

# Standard library imports
import json
import logging
from pathlib import Path
from typing import Any

# Local imports
from src.experiments.base.experiment import ExperimentSummarizer

log = logging.getLogger(__name__)

_RESULTS_FILENAME = "results.json"
_CONFIG_FILENAME = "experiment_configs.json"
_EXPERIMENT_TYPE = "standard_forecasting"


class StandardForecastingSummarizer(ExperimentSummarizer):
    """Summarise runs under ``experiments/standard_forecasting/``.

    Each completed run is expected to contain a ``results.json`` file with
    the structure written by ``scripts/experiments/sliding_window_eval.py``::

        {
          "model": "ttm",
          "mode": "fine-tuned",
          "dataset": "aleppo_2017",
          "timestamp": "2026-02-16T18:51:20.393668",
          "config": {"context_length": 512, "forecast_length": 96},
          "overall": {"rmse": 3.48, "mae": 2.70, "mape": 36.5, "mse": 12.1},
          "per_patient": [{"patient": "ale_102", "episodes": 6, ...}, ...]
        }

    Parameters
    ----------
    experiments_root:
        Path to the top-level ``experiments/`` directory.  Defaults to
        ``"experiments"`` (relative to the working directory).
    """

    def __init__(self, experiments_root: str | Path = "experiments") -> None:
        super().__init__(experiments_root, _EXPERIMENT_TYPE)

    # ------------------------------------------------------------------
    # ExperimentSummarizer contract
    # ------------------------------------------------------------------

    def _parse_run_dir(
        self,
        ctx_fh: str,
        model: str,
        run_dir: Path,
    ) -> dict[str, Any] | None:
        """Extract a flat summary row from a single run directory.

        Returns ``None`` when ``results.json`` is absent, empty, or malformed.
        """
        results_path = run_dir / _RESULTS_FILENAME
        if not results_path.exists():
            log.debug("No %s in %s — skipping", _RESULTS_FILENAME, run_dir)
            return None

        try:
            with results_path.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not load %s: %s", results_path, exc)
            return None

        overall = data.get("overall", {})
        if not overall:
            log.debug("Empty 'overall' block in %s — skipping", results_path)
            return None

        per_patient = data.get("per_patient", [])
        total_episodes = int(sum(p.get("episodes", 0) for p in per_patient))

        # Normalise mode string ("fine-tuned" → "finetuned")
        mode_raw: str = data.get("mode", "")
        mode = mode_raw.lower().replace("-", "")

        cfg = data.get("config", {})
        git_commit = _read_git_commit(run_dir)

        return {
            "run_id": run_dir.name,
            "experiment_type": _EXPERIMENT_TYPE,
            "ctx_fh": ctx_fh,
            "model": data.get("model", model),
            "dataset": data.get("dataset", ""),
            "mode": mode,
            "timestamp": data.get("timestamp", ""),
            "context_length": cfg.get("context_length"),
            "forecast_length": cfg.get("forecast_length"),
            "rmse": overall.get("rmse"),
            "mae": overall.get("mae"),
            "mape": overall.get("mape"),
            "mse": overall.get("mse"),
            "total_episodes": total_episodes,
            "git_commit": git_commit,
            "run_path": str(run_dir),
        }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _read_git_commit(run_dir: Path) -> str | None:
    """Try to extract the git commit SHA from ``experiment_configs.json``."""
    config_path = run_dir / _CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        with config_path.open() as fh:
            cfg = json.load(fh)
        return cfg.get("environment", {}).get("git_commit")
    except (json.JSONDecodeError, OSError):
        return None
