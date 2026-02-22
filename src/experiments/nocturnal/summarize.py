# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Summarizer for nocturnal hypoglycemia forecasting experiments.

Scans all run directories under ``experiments/nocturnal_forecasting/`` and
builds a tidy CSV with one row per run.  Extends
:class:`~src.experiments.base.experiment.ExperimentSummarizer` with logic to
parse ``nocturnal_results.json`` files as produced by
``scripts/experiments/nocturnal_hypo_eval.py``.

Note: nocturnal evaluation only computes RMSE at the overall level; MAE, MAPE,
and MSE are ``NaN`` in the summary (they are available per-patient only).

Module: experiments.nocturnal.summarize
Author: Blood-Glucose-Control
Created: 2026-02-22
"""

from __future__ import annotations

# Standard library imports
import json
import logging
import math
from pathlib import Path
from typing import Any

# Local imports
from src.experiments.base.experiment import ExperimentSummarizer

log = logging.getLogger(__name__)

_RESULTS_FILENAME = "nocturnal_results.json"
_CONFIG_FILENAME = "experiment_config.json"  # singular — nocturnal convention
_CONFIG_FILENAME_ALT = "experiment_configs.json"
_EXPERIMENT_TYPE = "nocturnal_forecasting"


class NocturnalSummarizer(ExperimentSummarizer):
    """Summarise runs under ``experiments/nocturnal_forecasting/``.

    Each completed run is expected to contain a ``nocturnal_results.json``
    file with the structure written by
    ``scripts/experiments/nocturnal_hypo_eval.py``::

        {
          "evaluation_type": "nocturnal_hypoglycemia",
          "model": "ttm",
          "mode": "zero-shot",
          "dataset": "brown_2019",
          "timestamp": "2026-02-21T22:43:03.959874",
          "config": {"context_length": 512, "forecast_length": 72},
          "overall_rmse": 3.53,
          "total_episodes": 1830,
          "per_patient": [{"patient_id": "bro_75", "episodes": 6, ...}, ...]
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
        """Extract a flat summary row from a single nocturnal run directory.

        Returns ``None`` when ``nocturnal_results.json`` is absent, empty, or
        malformed.
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

        overall_rmse = data.get("overall_rmse")
        if overall_rmse is None:
            log.debug("Missing 'overall_rmse' in %s — skipping", results_path)
            return None
        if isinstance(overall_rmse, float) and math.isnan(overall_rmse):
            log.debug("'overall_rmse' is NaN in %s — skipping", results_path)
            return None

        total_episodes = data.get("total_episodes")
        if total_episodes is None:
            # Fall back to summing per_patient episodes
            per_patient = data.get("per_patient", [])
            total_episodes = int(sum(p.get("episodes", 0) for p in per_patient))
        if int(total_episodes) == 0:
            log.debug("'total_episodes' is 0 in %s — skipping", results_path)
            return None

        # Normalise mode string ("zero-shot" → "zeroshot")
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
            "rmse": overall_rmse,
            # MAE / MAPE / MSE are not reported at the overall level for
            # nocturnal evaluation — set to NaN so the column schema is
            # consistent with StandardForecastingSummarizer.
            "mae": float("nan"),
            "mape": float("nan"),
            "mse": float("nan"),
            "total_episodes": int(total_episodes),
            "git_commit": git_commit,
            "run_path": str(run_dir),
        }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _read_git_commit(run_dir: Path) -> str | None:
    """Try to extract the git commit SHA from an experiment config file."""
    for fname in (_CONFIG_FILENAME, _CONFIG_FILENAME_ALT):
        config_path = run_dir / fname
        if config_path.exists():
            try:
                with config_path.open() as fh:
                    cfg = json.load(fh)
                return cfg.get("environment", {}).get("git_commit")
            except (json.JSONDecodeError, OSError):
                pass
    return None
