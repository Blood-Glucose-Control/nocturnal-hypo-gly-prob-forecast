# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Episode-level classification of "any-step hypoglycemia" from existing runs.

For each midnight-anchored episode we already store the actuals and the
quantile forecasts (Tier 3 in ``forecasts.npz``).  This module derives a
binary label and **two** continuous risk scores per episode:

  * label      ``y_ep      = 1{ min_t actuals[t] < HYPO_THRESHOLD_MMOL }``
  * score_max  ``s_ep^max  = max_t  P(BG_t < HYPO_THRESHOLD_MMOL)``
  * score_mean ``s_ep^mean = mean_t P(BG_t < HYPO_THRESHOLD_MMOL)``

then evaluates how well each score ranks the binary outcome with **AUROC**
and **AUPRC** pooled per run/split.  Metric columns are suffixed
``_max`` / ``_mean`` so both score variants appear in the same row of the
output table.  AUROC is base-rate invariant; AUPRC is reported alongside
its random-chance baseline (the base rate) and the skill score
``AUPRC - base_rate``.

Aggregation: pooled per (run × split_type).  No bootstrap CIs.

Models that don't emit quantile forecasts (e.g. TTM, deterministic
baselines) get ``NaN`` everywhere — the score is undefined for them.

Module: experiments.nocturnal.episode_classification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiments.nocturnal.grand_summary import (
    DATASETS,
    MODEL_PROPERTIES,
    attach_covariate_bucket,
    dedupe,
    load_summary,
    pick_best,
)
from src.experiments.nocturnal.grand_summary_split import (
    attach_config_dir,
    _model_class,
)
from src.experiments.nocturnal.holdout_split_analysis import (
    HYPO_THRESHOLD_MMOL,
    SPLIT_PATIENT,
    SPLIT_TEMPORAL,
    classify_episodes,
    load_holdout_patient_set,
)

log = logging.getLogger(__name__)

_SPLITS: tuple[str, ...] = (SPLIT_PATIENT, SPLIT_TEMPORAL)


# ---------------------------------------------------------------------------
# Per-step hypo probability from the quantile CDF
# ---------------------------------------------------------------------------


def per_step_hypo_probability(
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
    threshold: float = HYPO_THRESHOLD_MMOL,
) -> np.ndarray:
    """Estimate ``P(BG_t < threshold)`` at every (episode, step).

    For each (episode, step) the quantile values are sorted along the
    quantile axis and ``threshold`` is linearly interpolated against the
    discrete CDF (BG values → cumulative probability), exactly matching the
    semantics in :func:`src.evaluation.metrics.probabilistic.compute_brier_score`,
    including the conservative ``[q_min, q_max]`` clamping at the tails.

    Args:
        quantile_forecasts: shape ``(n_episodes, n_quantiles, forecast_length)``.
        quantile_levels:    shape ``(n_quantiles,)``, strictly increasing in
                            (0, 1).
        threshold:          BG threshold in mmol/L.

    Returns:
        Array of shape ``(n_episodes, forecast_length)`` of ``p_hat`` values
        in ``[q_min, q_max]``.
    """
    qf = np.asarray(quantile_forecasts, dtype=np.float64)
    q_arr = np.asarray(quantile_levels, dtype=np.float64)

    if qf.ndim != 3:
        raise ValueError(
            f"quantile_forecasts must be 3D (n_eps, n_q, fh); got {qf.shape}"
        )
    if q_arr.ndim != 1 or q_arr.size != qf.shape[1]:
        raise ValueError(
            f"quantile_levels {q_arr.shape} incompatible with q axis {qf.shape[1]}"
        )
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")

    n_eps, n_q, fh = qf.shape
    q_min, q_max = float(q_arr[0]), float(q_arr[-1])

    # Sort BG values per (episode, step) so the per-(ep, step) "xp" is
    # monotone-non-decreasing.
    xs = np.sort(qf, axis=1)  # (n_eps, n_q, fh)

    # Vectorised per-(ep, step) np.interp(threshold, xs, q_arr, left=q_min, right=q_max):
    # find insertion index along the quantile axis.
    idx = (xs < threshold).sum(axis=1)  # (n_eps, fh) ∈ [0, n_q]
    idx_low = np.clip(idx - 1, 0, n_q - 1)
    idx_high = np.clip(idx, 0, n_q - 1)

    x_low = np.take_along_axis(xs, idx_low[:, None, :], axis=1)[:, 0, :]
    x_high = np.take_along_axis(xs, idx_high[:, None, :], axis=1)[:, 0, :]
    y_low = q_arr[idx_low]
    y_high = q_arr[idx_high]

    denom = x_high - x_low
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom > 0, (threshold - x_low) / denom, 0.0)
    p_hat = y_low + frac * (y_high - y_low)
    # Tail clamping to match np.interp(..., left=q_min, right=q_max).
    p_hat = np.where(idx == 0, q_min, p_hat)
    p_hat = np.where(idx == n_q, q_max, p_hat)
    return p_hat


# ---------------------------------------------------------------------------
# Per-run loader specialised for classification (skips episode parquet read
# of arrays we don't need + reads quantile_forecasts from npz).
# ---------------------------------------------------------------------------


def _read_run_meta(run_path: Path) -> tuple[str, str] | tuple[None, None]:
    cfg_path = run_path / "experiment_config.json"
    if not cfg_path.exists():
        return None, None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    cli = cfg.get("cli_args", {}) or {}
    cd = cli.get("config_dir")
    ds = cli.get("dataset")
    return (str(cd) if cd else None, str(ds) if ds else None)


def load_run_classification_data(
    run_path: str | Path,
    config_dir: str | Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Load per-episode label + ``max_phat`` / ``mean_phat`` scores for one run.

    Returns a DataFrame with columns:
      ``patient_id, split_type, has_hypo, max_phat, mean_phat``

    Both score columns are ``NaN`` for runs without ``quantile_forecasts``
    (deterministic models).
    """
    run_path = Path(run_path)
    parquet_path = run_path / "episodes.parquet"
    npz_path = run_path / "forecasts.npz"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if config_dir is None or dataset is None:
        cd, ds = _read_run_meta(run_path)
        config_dir = config_dir or cd
        dataset = dataset or ds
        if config_dir is None or dataset is None:
            raise ValueError(
                f"{run_path} missing cli_args.config_dir/dataset; pass explicitly"
            )

    holdout_patients = load_holdout_patient_set(config_dir, dataset)

    parquet_df = pd.read_parquet(parquet_path, columns=["patient_id"])

    with np.load(npz_path, allow_pickle=False) as npz:
        actuals = np.asarray(npz["actuals"])
        q_forecasts = np.asarray(npz.get("quantile_forecasts", np.empty((0, 0, 0))))
        q_levels = np.asarray(npz.get("quantile_levels", np.empty(0)))

    if len(actuals) != len(parquet_df):
        raise ValueError(
            f"{run_path}: episodes.parquet has {len(parquet_df)} rows but "
            f"forecasts.npz has {len(actuals)} — files are out of sync."
        )

    has_hypo = actuals.min(axis=1) < HYPO_THRESHOLD_MMOL

    if q_forecasts.size == 0 or q_levels.size == 0:
        max_phat = np.full(len(parquet_df), np.nan, dtype=np.float64)
        mean_phat = np.full(len(parquet_df), np.nan, dtype=np.float64)
    else:
        if q_forecasts.shape[0] != len(parquet_df):
            raise ValueError(
                f"{run_path}: quantile_forecasts has {q_forecasts.shape[0]} "
                f"rows, expected {len(parquet_df)}"
            )
        p_step = per_step_hypo_probability(q_forecasts, q_levels)
        max_phat = p_step.max(axis=1)
        mean_phat = p_step.mean(axis=1)

    out = pd.DataFrame(
        {
            "patient_id": parquet_df["patient_id"].astype(str).to_numpy(),
            "split_type": classify_episodes(parquet_df["patient_id"], holdout_patients),
            "has_hypo": has_hypo,
            "max_phat": max_phat,
            "mean_phat": mean_phat,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Per-split classification metrics
# ---------------------------------------------------------------------------


def episode_classification_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Pooled classification metrics for a set of episodes.

    Returns ``base_rate``, ``auroc``, ``auprc``, ``auprc_skill``,
    ``n_episodes``, ``n_pos``, ``n_neg``.

    ``auroc``/``auprc`` are ``NaN`` when scores are all-NaN (deterministic
    model) or when one class is missing from ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)

    n = int(y_true.size)
    n_pos = int(y_true.sum())
    n_neg = int(n - n_pos)
    base_rate = float(n_pos / n) if n else float("nan")

    out = {
        "n_episodes": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": base_rate,
        "auroc": float("nan"),
        "auprc": float("nan"),
        "auprc_skill": float("nan"),
    }
    if n_pos == 0 or n_neg == 0:
        return out
    valid = ~np.isnan(scores)
    if not valid.any():
        return out
    y_v = y_true[valid]
    s_v = scores[valid]
    if y_v.sum() == 0 or y_v.sum() == y_v.size:
        return out
    out["auroc"] = float(roc_auc_score(y_v, s_v))
    out["auprc"] = float(average_precision_score(y_v, s_v))
    # Skill vs always predicting the marginal: random AUPRC == base rate.
    # Use the *valid-subset* base rate so skill is internally consistent.
    sub_rate = float(y_v.mean())
    out["auprc_skill"] = out["auprc"] - sub_rate
    return out


# ---------------------------------------------------------------------------
# Per-run aggregation
# ---------------------------------------------------------------------------


def aggregate_run_classification(
    run_path: str | Path,
    *,
    model: str,
    dataset: str,
    cov_bucket: str,
    config_dir: str,
) -> list[dict]:
    """One row per ``split_type`` of classification metrics for one run.

    Each row carries both ``*_max`` and ``*_mean`` variants of AUROC, AUPRC,
    and AUPRC skill so the two episode-score definitions can be compared
    side-by-side. Counts (``n_episodes``, ``n_pos``, ``n_neg``,
    ``base_rate``) are score-independent and reported once.
    """
    df = load_run_classification_data(run_path, config_dir=config_dir, dataset=dataset)
    rows: list[dict] = []
    for split in _SPLITS:
        sub = df[df["split_type"] == split]
        y = sub["has_hypo"].to_numpy()
        m_max = episode_classification_metrics(y, sub["max_phat"].to_numpy())
        m_mean = episode_classification_metrics(y, sub["mean_phat"].to_numpy())
        shared = {k: m_max[k] for k in ("n_episodes", "n_pos", "n_neg", "base_rate")}
        scored = {
            "auroc_max": m_max["auroc"],
            "auroc_mean": m_mean["auroc"],
            "auprc_max": m_max["auprc"],
            "auprc_mean": m_mean["auprc"],
            "auprc_skill_max": m_max["auprc_skill"],
            "auprc_skill_mean": m_mean["auprc_skill"],
        }
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "cov_bucket": cov_bucket,
                "split_type": split,
                "n_patients": int(sub["patient_id"].nunique()),
                **shared,
                **scored,
                "run_path": str(run_path),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Top-level pipeline (mirrors build_best_split_breakdown)
# ---------------------------------------------------------------------------


def build_best_episode_classification(
    summary_paths: Sequence[str | Path],
    config_dir: str,
    *,
    datasets: Sequence[str] = DATASETS,
    ctx_filter: int | None = None,
    forecast_filter: int | None = 96,
) -> pd.DataFrame:
    """Build the long-form best-runs episode-classification table.

    Filters, dedupes, buckets, and picks best runs the same way as
    :func:`grand_summary_split.build_best_split_breakdown`, then computes
    AUROC/AUPRC for the binary "any-step hypo" task per (run × split).
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
    best = pick_best(bucketed)

    rows: list[dict] = []
    for _, r in best.iterrows():
        run_path = str(r["run_path"])
        try:
            split_rows = aggregate_run_classification(
                run_path,
                model=str(r["model"]),
                dataset=str(r["dataset"]),
                cov_bucket=str(r["cov_bucket"]),
                config_dir=config_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping %s: %s", run_path, exc)
            continue
        for sr in split_rows:
            sr["model_class"] = _model_class(sr["model"])
            sr["context_length"] = r.get("context_length")
            sr["forecast_length"] = r.get("forecast_length")
            rows.append(sr)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["model_class", "model", "dataset", "cov_bucket", "split_type"]
    ).reset_index(drop=True)
    lead = [
        "model_class",
        "model",
        "dataset",
        "cov_bucket",
        "split_type",
        "n_patients",
        "n_episodes",
        "n_pos",
        "n_neg",
        "base_rate",
        "auroc_max",
        "auroc_mean",
        "auprc_max",
        "auprc_mean",
        "auprc_skill_max",
        "auprc_skill_mean",
    ]
    rest = [c for c in out.columns if c not in lead]
    out = out[[c for c in lead if c in out.columns] + rest]
    return out


__all__ = [
    "aggregate_run_classification",
    "build_best_episode_classification",
    "episode_classification_metrics",
    "load_run_classification_data",
    "per_step_hypo_probability",
]
