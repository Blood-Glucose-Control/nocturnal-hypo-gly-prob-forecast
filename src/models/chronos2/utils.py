# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""Chronos-2 compatibility helpers used by tests and scratch experiments.

This module intentionally exposes only the midnight episode builder. Legacy
evaluation/plotting helpers were retired after no-caller verification in
maintained runtime/workflow paths.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...evaluation.episode_builders import (
    build_midnight_episodes as _build_midnight_episodes_shared,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_midnight_episodes",
]


def build_midnight_episodes(
    patient_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = 5,
    context_len: int = 512,
    horizon: int = 72,
) -> List[Dict[str, Any]]:
    """Build Chronos-2 style midnight episodes with covariate compatibility.

    The shared evaluation builder is now the source of truth. This wrapper keeps
    Chronos-2 compatibility behavior:
    - defaults covariates to ["iob"],
    - returns [] when no requested covariates are available,
    - ensures each episode includes `covariates_at_midnight`.
    """
    if covariate_cols is None:
        covariate_cols = ["iob"]

    df = patient_df.sort_index()
    available_covs = [
        c for c in covariate_cols if c in df.columns and df[c].notna().any()
    ]
    if not available_covs:
        logger.warning("No covariate data available (need one of %s)", covariate_cols)
        return []

    episodes, _ = _build_midnight_episodes_shared(
        patient_df=df,
        context_length=context_len,
        forecast_length=horizon,
        target_col=target_col,
        covariate_cols=covariate_cols,
        interval_mins=interval_mins,
    )

    for episode in episodes:
        context_df = episode["context_df"].copy()
        future_covariates = episode.setdefault("future_covariates", {})
        covariates_at_midnight: Dict[str, float] = {}

        for cov_col in covariate_cols:
            if cov_col in context_df.columns:
                context_df[cov_col] = context_df[cov_col].ffill().fillna(0)
                covariates_at_midnight[cov_col] = float(context_df[cov_col].iloc[-1])
            else:
                context_df[cov_col] = 0.0
                covariates_at_midnight[cov_col] = 0.0

            if cov_col not in future_covariates:
                future_covariates[cov_col] = np.zeros(horizon)
            else:
                vals = np.asarray(future_covariates[cov_col], dtype=np.float32)
                if len(vals) < horizon:
                    padded = np.zeros(horizon, dtype=np.float32)
                    padded[: len(vals)] = vals
                    vals = padded
                future_covariates[cov_col] = (
                    pd.Series(vals[:horizon]).ffill().fillna(0).to_numpy()
                )

        episode["context_df"] = context_df
        episode["future_covariates"] = future_covariates
        episode["covariates_at_midnight"] = covariates_at_midnight

    return episodes
