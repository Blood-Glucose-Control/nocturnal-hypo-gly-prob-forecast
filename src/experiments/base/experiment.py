# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Base class for experiment summarizers.

Provides generic directory-walking and CSV aggregation logic that concrete
experiment summarizers (standard_forecasting, nocturnal_forecasting, …) build
on by implementing ``_parse_run_dir``.

Module: experiments.base.experiment
Author: Christopher Risi
Created: 2026-02-22
"""

from __future__ import annotations

# Standard library imports
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Third-party imports
import pandas as pd

log = logging.getLogger(__name__)

# Run-directory naming convention: YYYY-MM-DD_HHMM_<dataset>_<mode>
_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_\S+$")

VALID_METRICS = {"rmse", "mae", "mape", "mse"}


class ExperimentSummarizer(ABC):
    """Walk an experiment tree, produce a tidy CSV, and surface best runs.

    Directory layout expected::

        <experiments_root>/
          <experiment_type>/          e.g. standard_forecasting
            <ctx_fh>/                 e.g. 512ctx_96fh
              <model>/                e.g. ttm, sundial
                <run_dir>/           e.g. 2026-02-16_1808_aleppo_2017_finetuned

    Parameters
    ----------
    experiments_root:
        Absolute or relative path to the top-level ``experiments/`` folder.
    experiment_type:
        Name of the experiment subdirectory (e.g. ``"standard_forecasting"``).
    """

    def __init__(self, experiments_root: str | Path, experiment_type: str) -> None:
        self.experiments_root = Path(experiments_root)
        self.experiment_type = experiment_type
        self.experiment_dir = self.experiments_root / experiment_type

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_run_dirs(self):
        """Yield ``(ctx_fh, model, run_dir_path)`` for every completed run.

        A valid run directory is any leaf directory whose name matches the
        ``YYYY-MM-DD_HHMM_*`` convention.
        """
        if not self.experiment_dir.is_dir():
            log.warning("Experiment directory not found: %s", self.experiment_dir)
            return

        for ctx_fh_dir in sorted(self.experiment_dir.iterdir()):
            if not ctx_fh_dir.is_dir():
                continue
            for model_dir in sorted(ctx_fh_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for run_dir in sorted(model_dir.iterdir()):
                    if run_dir.is_dir() and _RUN_DIR_RE.match(run_dir.name):
                        yield ctx_fh_dir.name, model_dir.name, run_dir

    @abstractmethod
    def _parse_run_dir(
        self,
        ctx_fh: str,
        model: str,
        run_dir: Path,
    ) -> dict[str, Any] | None:
        """Extract a flat dict of fields from *run_dir*.

        Returns ``None`` for incomplete or unreadable runs (they will be
        skipped silently after a ``DEBUG`` log message).
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        metric: str = "rmse",
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Scan all run directories and return a tidy summary ``DataFrame``.

        The DataFrame is also written to ``summary.csv`` inside the experiment
        directory (or *output_path* if provided).

        Parameters
        ----------
        metric:
            Metric used only for column ordering; does not filter rows.
        output_path:
            Override destination for ``summary.csv``.

        Returns
        -------
        pd.DataFrame
            One row per run, sorted by ``(model, dataset, <metric>)`` where the
            metric column is present.
        """
        _validate_metric(metric)
        rows: list[dict[str, Any]] = []
        for ctx_fh, model, run_dir in self._iter_run_dirs():
            row = self._parse_run_dir(ctx_fh, model, run_dir)
            if row is None:
                log.debug("Skipping incomplete run: %s", run_dir)
                continue
            rows.append(row)

        if not rows:
            log.warning("No completed runs found under %s", self.experiment_dir)
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Sort so the primary ranking metric is visible early
        sort_cols = ["model", "dataset"]
        if metric in df.columns:
            sort_cols.append(metric)
        df = df.sort_values(sort_cols, ignore_index=True)

        dest = Path(output_path) if output_path else self.experiment_dir / "summary.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        log.info("Summary written to %s  (%d runs)", dest, len(df))
        return df

    def best_runs(
        self,
        metric: str = "rmse",
    ) -> dict[str, pd.DataFrame]:
        """Return (and save) the best run per model and per (model × dataset).

        Calls :meth:`summarize` internally if not already run.

        Parameters
        ----------
        metric:
            Column name used to rank runs.  Must be numeric; lower is better.

        Returns
        -------
        dict with two keys:

        ``"by_model_dataset"``
            Best run for every ``(model, dataset)`` pair.
        ``"by_model"``
            Global best run for every model across all datasets.
        """
        _validate_metric(metric)
        df = self.summarize(metric=metric)

        if df.empty or metric not in df.columns:
            log.warning(
                "Cannot compute best runs: summary is empty or missing column '%s'.",
                metric,
            )
            return {"by_model_dataset": pd.DataFrame(), "by_model": pd.DataFrame()}

        if df[metric].isna().all():
            log.warning(
                "Metric '%s' is NaN for all runs in '%s' — cannot rank. "
                "This experiment type may not report '%s' at the overall level. "
                "Try --metric rmse instead.",
                metric,
                self.experiment_type,
                metric,
            )
            return {"by_model_dataset": pd.DataFrame(), "by_model": pd.DataFrame()}

        # Best per (model × dataset) — dropna guards against groups where the
        # chosen metric is entirely NaN (e.g. nocturnal runs have no overall MAE).
        by_md_idx = df.groupby(["model", "dataset"])[metric].idxmin().dropna()
        by_md = (
            df.loc[by_md_idx].sort_values(["model", "dataset"]).reset_index(drop=True)
        )

        # Global best per model
        by_m_idx = df.groupby("model")[metric].idxmin().dropna()
        by_m = df.loc[by_m_idx].sort_values("model").reset_index(drop=True)

        out_dir = self.experiment_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        by_md.to_csv(out_dir / "best_by_model_dataset.csv", index=False)
        by_m.to_csv(out_dir / "best_by_model.csv", index=False)
        log.info(
            "Best-run tables written to %s  (by_model_dataset: %d rows, by_model: %d rows)",
            out_dir,
            len(by_md),
            len(by_m),
        )

        return {"by_model_dataset": by_md, "by_model": by_m}


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _validate_metric(metric: str) -> None:
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Valid choices: {sorted(VALID_METRICS)}"
        )
