# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Helpers for splitting nocturnal evaluation episodes into the
patient-holdout vs temporal-holdout subsets used by hybrid data ablations.

A run's three-tier outputs (``episodes.parquet`` + ``forecasts.npz`` +
``experiment_config.json``) carry everything needed to label each midnight
episode by *which* type of holdout it came from:

  * The run's ``experiment_config.json::cli_args.config_dir`` points to the
    holdout YAML directory (e.g. ``configs/data/holdout_10pct``).
  * That directory contains ``<dataset>.yaml`` whose
    ``patient_config.holdout_patients`` lists the entirely-held-out patients.
  * Any episode whose ``patient_id`` is in that list belongs to the
    *patient* split; everything else is *temporal*.

The actuals array in ``forecasts.npz`` is row-aligned with
``episodes.parquet`` (both built from ``all_episode_results`` in
``src/evaluation/nocturnal.py``), so we can also flag any episode containing
a true BG reading below the hypoglycemia threshold (3.9 mmol/L).

Module: experiments.nocturnal.holdout_split_analysis
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.versioning.holdout_config import HoldoutConfig

log = logging.getLogger(__name__)

# Matches HYPO_THRESHOLD_MMOL in src/evaluation/nocturnal.py.
HYPO_THRESHOLD_MMOL: float = 3.9

SPLIT_PATIENT: str = "patient"
SPLIT_TEMPORAL: str = "temporal"


# ---------------------------------------------------------------------------
# Holdout YAML helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_holdout_patient_set(config_dir: str | Path, dataset: str) -> frozenset[str]:
    """Return the set of patient IDs held out *entirely* for ``dataset``.

    Reads ``<config_dir>/<dataset>.yaml`` and pulls
    ``patient_config.holdout_patients``. Cached per ``(config_dir, dataset)``.
    """
    yaml_path = Path(config_dir) / f"{dataset}.yaml"
    cfg = HoldoutConfig.load(yaml_path)
    if cfg.patient_config is None or not cfg.patient_config.holdout_patients:
        log.warning(
            "%s has no patient_config.holdout_patients — all episodes will "
            "be classified as temporal-holdout.",
            yaml_path,
        )
        return frozenset()
    return frozenset(str(p) for p in cfg.patient_config.holdout_patients)


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def read_run_config_dir(run_path: str | Path) -> str | None:
    """Return the holdout ``config_dir`` recorded in a run's experiment config.

    None if the file is missing or the field is absent.
    """
    cfg_path = Path(run_path) / "experiment_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", cfg_path, exc)
        return None
    cli = cfg.get("cli_args", {}) or {}
    cd = cli.get("config_dir")
    return str(cd) if cd else None


def read_run_dataset(run_path: str | Path) -> str | None:
    """Return the dataset name recorded in a run's experiment config."""
    cfg_path = Path(run_path) / "experiment_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", cfg_path, exc)
        return None
    cli = cfg.get("cli_args", {}) or {}
    ds = cli.get("dataset")
    return str(ds) if ds else None


# ---------------------------------------------------------------------------
# Episode classification
# ---------------------------------------------------------------------------


def classify_episodes(
    patient_ids: Iterable[str],
    holdout_patient_set: Iterable[str],
) -> np.ndarray:
    """Return an array of ``"patient"``/``"temporal"`` per episode."""
    holdout = frozenset(str(p) for p in holdout_patient_set)
    return np.array(
        [SPLIT_PATIENT if str(p) in holdout else SPLIT_TEMPORAL for p in patient_ids],
        dtype=object,
    )


def episode_has_hypo(
    actuals: np.ndarray,
    threshold: float = HYPO_THRESHOLD_MMOL,
) -> np.ndarray:
    """Boolean array, one per episode, True iff any forecast-window step
    has an actual BG reading strictly below ``threshold``.
    """
    if actuals.ndim != 2:
        raise ValueError(
            f"Expected actuals of shape (n_episodes, fh); got {actuals.shape}"
        )
    return actuals.min(axis=1) < threshold


# ---------------------------------------------------------------------------
# Per-run join
# ---------------------------------------------------------------------------


def load_run_episode_classification(
    run_path: str | Path,
    config_dir: str | Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Load a run's per-episode metrics, classified by holdout split.

    Returns the contents of ``episodes.parquet`` augmented with:

      * ``split_type``  – ``"patient"`` or ``"temporal"``
      * ``has_hypo``    – True iff any actual BG step in the forecast window
                          is below ``HYPO_THRESHOLD_MMOL``
      * ``actual_min``  – the minimum actual BG in the forecast window
      * ``predictions`` – per-episode point forecast array (object dtype)
      * ``actuals``     – per-episode actuals array (object dtype)

    ``config_dir`` and ``dataset`` default to the values recorded in the
    run's ``experiment_config.json``.
    """
    run_path = Path(run_path)
    parquet_path = run_path / "episodes.parquet"
    npz_path = run_path / "forecasts.npz"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    df = pd.read_parquet(parquet_path)

    if config_dir is None:
        config_dir = read_run_config_dir(run_path)
        if config_dir is None:
            raise ValueError(
                f"{run_path} has no cli_args.config_dir; pass config_dir explicitly"
            )
    if dataset is None:
        dataset = read_run_dataset(run_path)
        if dataset is None:
            raise ValueError(
                f"{run_path} has no cli_args.dataset; pass dataset explicitly"
            )

    holdout_patients = load_holdout_patient_set(config_dir, dataset)

    with np.load(npz_path, allow_pickle=False) as npz:
        actuals = np.asarray(npz["actuals"])
        predictions = np.asarray(npz["predictions"])
        episode_ids = np.asarray(npz["episode_ids"])

    if len(actuals) != len(df):
        # Row-alignment is enforced by the storage layer (single iteration
        # over all_episode_results in src/evaluation/nocturnal.py); a
        # mismatch indicates a corrupted / mixed-source run directory.
        raise ValueError(
            f"{run_path}: episodes.parquet has {len(df)} rows but "
            f"forecasts.npz has {len(actuals)} — files are out of sync."
        )

    # Defensive: cross-check via reconstructed episode_id (matches the
    # f"{patient_id}::ep{i:03d}" format used in evaluate_nocturnal_forecasting).
    # We don't fail on mismatch (anchor ordering across patients is implicit
    # in the for-loop iteration order) — only the row count is contractual.
    if len(episode_ids) and "patient_id" in df.columns:
        # Cheap sanity: every parquet patient_id should be a prefix of the
        # corresponding episode_id string.
        prefix_ok = np.array(
            [
                str(eid).startswith(f"{pid}::")
                for eid, pid in zip(episode_ids, df["patient_id"].astype(str))
            ]
        )
        if not prefix_ok.all():
            n_bad = int((~prefix_ok).sum())
            log.warning(
                "%s: %d/%d episodes have parquet.patient_id not matching "
                "the npz episode_id prefix; alignment may be wrong.",
                run_path,
                n_bad,
                len(prefix_ok),
            )

    df = df.copy()
    df["split_type"] = classify_episodes(df["patient_id"], holdout_patients)
    df["actual_min"] = actuals.min(axis=1)
    df["has_hypo"] = df["actual_min"] < HYPO_THRESHOLD_MMOL
    # Stash arrays for downstream re-aggregation (e.g. split-RMSE).
    df["predictions"] = list(predictions)
    df["actuals"] = list(actuals)
    return df


# ---------------------------------------------------------------------------
# Cohort statistics from raw data (incl. training cohort)
# ---------------------------------------------------------------------------

# Cohort labels for the three-way split:
#   - patient_holdout : entirely held-out patients (in YAML's holdout_patients)
#   - temporal_holdout: tail-end of non-holdout patients (the temporal split)
#   - temporal_train  : everything else (the training data)
COHORT_PATIENT_HOLDOUT: str = "patient_holdout"
COHORT_TEMPORAL_HOLDOUT: str = "temporal_holdout"
COHORT_TEMPORAL_TRAIN: str = "temporal_train"


def _patient_episode_stats(
    patient_df: pd.DataFrame,
    *,
    context_length: int,
    forecast_length: int,
    target_col: str,
    interval_mins: int,
) -> tuple[int, list[float], list[float]]:
    """Build midnight episodes for a single patient and return
    ``(n_episodes, list_of_actuals_min, list_of_actuals_flat)``.

    Episodes that don't yield a full ``forecast_length`` window are skipped,
    matching the filter applied in :func:`evaluate_nocturnal_forecasting`.
    """
    # Local import: episode_builders pulls numba which is heavy.
    from src.evaluation.episode_builders import build_midnight_episodes

    pdf = patient_df.copy()
    if not isinstance(pdf.index, pd.DatetimeIndex):
        if "datetime" in pdf.columns:
            pdf["datetime"] = pd.to_datetime(pdf["datetime"])
            pdf = pdf.set_index("datetime").sort_index()
        else:
            return 0, [], []

    episodes, _ = build_midnight_episodes(
        pdf,
        context_length=context_length,
        forecast_length=forecast_length,
        target_col=target_col,
        covariate_cols=None,
        interval_mins=interval_mins,
    )
    mins: list[float] = []
    flat: list[float] = []
    n = 0
    for ep in episodes:
        tgt = np.asarray(ep["target_bg"])
        if len(tgt) < forecast_length:
            continue
        n += 1
        mins.append(float(np.min(tgt)))
        flat.append(float(np.mean(tgt)))  # not used directly; kept for parity
    # For BG mean/std we want the raw episode arrays, not just the per-episode
    # mean — return them flattened too.
    return n, mins, flat


def cohort_summary(
    cohort_df: pd.DataFrame,
    *,
    cohort_label: str,
    dataset: str,
    context_length: int = 512,
    forecast_length: int = 96,
    target_col: str = "bg_mM",
    interval_mins: int = 5,
    patient_col: str = "p_num",
) -> dict:
    """Compute the per-cohort summary stats requested for the variance check.

    Returns a dict with keys: ``cohort, dataset, n_patients, n_rows,
    n_episodes, n_episodes_with_hypo, hypo_rate, mean_bg, std_bg, mean_min_bg``.

    BG mean/std are computed on the raw ``target_col`` values across the
    cohort (one observation per row of ``cohort_df``); ``mean_min_bg`` is
    the mean of per-episode forecast-window minima, matching how the
    headline RMSE table is built.
    """
    if cohort_df.empty:
        return {
            "cohort": cohort_label,
            "dataset": dataset,
            "n_patients": 0,
            "n_rows": 0,
            "n_episodes": 0,
            "n_episodes_with_hypo": 0,
            "hypo_rate": float("nan"),
            "mean_bg": float("nan"),
            "std_bg": float("nan"),
            "mean_min_bg": float("nan"),
        }

    bg = pd.to_numeric(cohort_df[target_col], errors="coerce")
    n_rows = int(bg.notna().sum())
    mean_bg = float(bg.mean(skipna=True))
    std_bg = float(bg.std(skipna=True))
    n_patients = int(cohort_df[patient_col].nunique())

    n_episodes_total = 0
    all_mins: list[float] = []
    for pid, pdf in cohort_df.groupby(patient_col):
        n_eps, mins, _ = _patient_episode_stats(
            pdf,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            interval_mins=interval_mins,
        )
        n_episodes_total += n_eps
        all_mins.extend(mins)

    arr = np.asarray(all_mins) if all_mins else np.empty(0)
    n_hypo = int(np.sum(arr < HYPO_THRESHOLD_MMOL)) if arr.size else 0
    return {
        "cohort": cohort_label,
        "dataset": dataset,
        "n_patients": n_patients,
        "n_rows": n_rows,
        "n_episodes": n_episodes_total,
        "n_episodes_with_hypo": n_hypo,
        "hypo_rate": (n_hypo / n_episodes_total) if n_episodes_total else float("nan"),
        "mean_bg": mean_bg,
        "std_bg": std_bg,
        "mean_min_bg": float(arr.mean()) if arr.size else float("nan"),
    }


def build_three_way_cohort_stats(
    config_dir: str | Path,
    dataset: str,
    *,
    context_length: int = 512,
    forecast_length: int = 96,
    target_col: str = "bg_mM",
    interval_mins: int = 5,
    patient_col: str = "p_num",
) -> list[dict]:
    """Compute cohort stats for ``patient_holdout``, ``temporal_holdout``,
    and ``temporal_train`` for one dataset.

    Loads the dataset via :class:`DatasetRegistry` using the holdout config at
    ``<config_dir>/<dataset>.yaml``, then partitions:

      * train_data                          → ``temporal_train``
      * holdout_data ∩ YAML.holdout_patients → ``patient_holdout``
      * holdout_data \\ YAML.holdout_patients → ``temporal_holdout``
    """
    from src.data.versioning.dataset_registry import DatasetRegistry

    registry = DatasetRegistry(holdout_config_dir=Path(config_dir))
    train_data, holdout_data = registry.load_dataset_with_split(
        dataset, patient_col=patient_col
    )
    holdout_patients = load_holdout_patient_set(config_dir, dataset)

    h_pat = holdout_data[holdout_data[patient_col].astype(str).isin(holdout_patients)]
    h_tmp = holdout_data[~holdout_data[patient_col].astype(str).isin(holdout_patients)]

    return [
        cohort_summary(
            h_pat,
            cohort_label=COHORT_PATIENT_HOLDOUT,
            dataset=dataset,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            interval_mins=interval_mins,
            patient_col=patient_col,
        ),
        cohort_summary(
            h_tmp,
            cohort_label=COHORT_TEMPORAL_HOLDOUT,
            dataset=dataset,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            interval_mins=interval_mins,
            patient_col=patient_col,
        ),
        cohort_summary(
            train_data,
            cohort_label=COHORT_TEMPORAL_TRAIN,
            dataset=dataset,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            interval_mins=interval_mins,
            patient_col=patient_col,
        ),
    ]
