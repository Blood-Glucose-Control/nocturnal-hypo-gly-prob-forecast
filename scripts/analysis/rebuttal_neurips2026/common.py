# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Shared utilities for the NeurIPS 2026 E&D rebuttal analyses (submission #3091).

All rebuttal analyses (A1-A6) are pure post-hoc processing of the per-episode
nocturnal-evaluation artifacts already on disk -- no model loading required.

Each completed run directory under
    experiments/nocturnal_forecasting/512ctx_96fh/<model>/<run>/
contains:
    - episodes.parquet : one row per midnight-anchored episode with per-episode
      metrics (rmse, wql, brier, dilate/shape/temporal, coverage/sharpness).
    - forecasts.npz    : raw arrays, row-aligned to episodes.parquet:
        predictions        (N, H)      point forecast (mmol/L)
        actuals            (N, H)       ground truth (mmol/L)
        episode_ids        (N,)         "<patient_id>::ep<NNN>"
        quantile_forecasts (N, Q, H)    quantile forecasts
        quantile_levels    (Q,)         e.g. [0.1, ..., 0.9]

Row alignment between the parquet and the npz is contractual (verified: the
per-episode RMSE recomputed from the npz arrays matches the parquet ``rmse``
column exactly). The patient/temporal holdout split is derived from the same
YAML configs used for training/eval (``configs/data/holdout_10pct``).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# NOTE: file lives at scripts/analysis/rebuttal_neurips2026/common.py, so the
# repo root is parents[3] (rebuttal_neurips2026 -> analysis -> scripts -> repo).
REPO = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO / "experiments" / "nocturnal_forecasting" / "512ctx_96fh"
HOLDOUT_DIR = REPO / "configs" / "data" / "holdout_10pct"

HYPO_MMOL: float = 3.9  # clinical nocturnal-hypoglycemia threshold
STEPS_PER_HOUR = 12  # 5-min cadence

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]
DATASET_LABEL = {
    "aleppo_2017": "Replace-BG",
    "brown_2019": "DCLP3",
    "lynch_2022": "IOBP2",
    "tamborlane_2008": "Tamborlane",
}


# --------------------------------------------------------------------------
# Holdout split
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def holdout_patients(
    dataset: str, config_dir: str = str(HOLDOUT_DIR)
) -> frozenset[str]:
    """Return the set of patient IDs assigned to the *patient* holdout."""
    cfg = yaml.safe_load((Path(config_dir) / f"{dataset}.yaml").read_text())
    return frozenset(cfg["patient_config"]["holdout_patients"])


def assign_split(dataset: str, patient_ids: np.ndarray) -> np.ndarray:
    """Vectorized patient/temporal split labels for an array of patient IDs."""
    held = holdout_patients(dataset)
    return np.where(np.isin(patient_ids, list(held)), "patient", "temporal")


# --------------------------------------------------------------------------
# Run loading
# --------------------------------------------------------------------------
@dataclass
class Run:
    model: str
    dataset: str
    mode: str
    run_path: Path
    episodes: pd.DataFrame  # per-episode metrics + pid/split/has_hypo/scores
    predictions: np.ndarray  # (N, H)
    actuals: np.ndarray  # (N, H)
    quantile_forecasts: np.ndarray | None  # (N, Q, H) or None
    quantile_levels: np.ndarray | None  # (Q,) or None


def _p_hypo_by_step(
    qf: np.ndarray, qlevels: np.ndarray, thr: float = HYPO_MMOL
) -> np.ndarray:
    """P(BG < thr) per step from a quantile forecast, via monotone-CDF interp.

    qf: (Q, H) quantile values; qlevels: (Q,) ascending probabilities.
    Sorts the BG values (x-axis) per step while keeping qlevels in ascending
    order so the implied CDF is monotone (handles mild quantile crossings).
    Clamps to [qlevels[0], qlevels[-1]] outside the quantile support.
    """
    H = qf.shape[1]
    out = np.empty(H, dtype=np.float64)
    for t in range(H):
        x = np.sort(qf[:, t])
        out[t] = np.interp(thr, x, qlevels, left=qlevels[0], right=qlevels[-1])
    return out


def load_run(
    run_path: str | Path,
    dataset: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    verify: bool = False,
    compute_phypo: bool = True,
) -> Run:
    """Load a single run's per-episode artifacts into a :class:`Run`.

    ``compute_phypo=False`` skips the per-episode P(BG<3.9) quantile interpolation
    (only needed by A2 --score prob and A6); this is a large speedup for analyses
    that use RMSE/WQL/point-score only (A1, A4, A7, A8).
    """
    run_path = Path(run_path)
    if not run_path.is_absolute():
        run_path = REPO / run_path
    df = pd.read_parquet(run_path / "episodes.parquet").reset_index(drop=True)
    z = np.load(run_path / "forecasts.npz", allow_pickle=True)

    pred = z["predictions"].astype(np.float64)
    act = z["actuals"].astype(np.float64)
    eids = np.asarray([str(e) for e in z["episode_ids"]])
    assert len(df) == len(pred) == len(eids), "parquet/npz length mismatch"

    if verify:
        rmse_npz = np.sqrt(((pred - act) ** 2).mean(axis=1))
        assert np.allclose(
            rmse_npz, df["rmse"].to_numpy(), atol=1e-6
        ), "row misalignment"

    pid = np.asarray([e.split("::")[0] for e in eids])
    if dataset is None:
        dataset = _infer_dataset(run_path)
    split = assign_split(dataset, pid)

    df = df.copy()
    df["episode_id"] = eids
    df["pid"] = pid
    df["split"] = split
    df["has_hypo"] = (act < HYPO_MMOL).any(axis=1)
    df["min_actual"] = act.min(axis=1)
    df["min_pred"] = pred.min(axis=1)
    df["score_point"] = -pred.min(axis=1)  # higher => more hypo risk

    qf = (
        z["quantile_forecasts"].astype(np.float64)
        if "quantile_forecasts" in z.files
        else None
    )
    ql = (
        z["quantile_levels"].astype(np.float64)
        if "quantile_levels" in z.files
        else None
    )
    # Point-only models (e.g. TTM) store empty (0,0,0) quantile arrays; guard the
    # empty / row-misaligned case and fall back to NaN probabilistic scores.
    has_q = (
        qf is not None
        and ql is not None
        and qf.ndim == 3
        and qf.size > 0
        and qf.shape[0] == len(df)
    )
    if has_q and compute_phypo:
        p_max = np.array([_p_hypo_by_step(qf[i], ql).max() for i in range(len(qf))])
        df["p_hypo_max"] = p_max
        df["score_prob"] = p_max
    else:
        if not has_q:
            qf = ql = None
        df["p_hypo_max"] = np.nan
        df["score_prob"] = np.nan

    return Run(
        model=model or _infer_model(run_path),
        dataset=dataset,
        mode=mode or "",
        run_path=run_path,
        episodes=df,
        predictions=pred,
        actuals=act,
        quantile_forecasts=qf,
        quantile_levels=ql,
    )


def _infer_dataset(run_path: Path) -> str:
    for d in DATASETS:
        if d in run_path.name:
            return d
    raise ValueError(f"cannot infer dataset from {run_path}")


def _infer_model(run_path: Path) -> str:
    return run_path.parent.name


# --------------------------------------------------------------------------
# Run discovery + selection (the FILESYSTEM is the source of truth)
# --------------------------------------------------------------------------
# IMPORTANT: summary.csv / best_by_model_dataset.csv are STALE and are treated
# as read-only historical records -- never read for selection, never written.
# Selection scans the run directories directly, reading each run's
# experiment_config.json (covariate condition + mode) and results_summary.json
# (overall_rmse, so best-condition selection reproduces the paper's table cells).
#
# Rerun-safe: Workstream-2 gap-fill runs keep landing (bg_only + zero-shot) and
# some arrive as ``*_rerun01`` collisions; scan_runs() picks them up on every
# call and best-selection breaks ties by lowest overall_rmse then newest mtime.

_SUMMARY_FILES = ("results_summary.json", "nocturnal_results.json")


def condition_bucket(covariate_cols) -> str:
    """Map covariate_cols -> {bg_only, iob, iob_cob} (the paper's G / I / IC).

    On-disk values seen: None -> bg_only; ['iob'] / ['iob','insulin_availability']
    -> iob; ['iob','cob'] -> iob_cob. ``carb_availability`` (a COB proxy) also
    routes to iob_cob.
    """
    if not covariate_cols:
        return "bg_only"
    s = {str(c).lower() for c in covariate_cols}
    if any(("cob" in c) or ("carb" in c) for c in s):
        return "iob_cob"
    if any(("iob" in c) or ("insulin" in c) for c in s):
        return "iob"
    return "bg_only"


def _mode_from_name(name: str) -> str:
    """Infer zero-shot vs fine-tuned from the run dir name (checkpoint flag is
    unreliable -- it is set even for naive/statistical runs)."""
    n = name.lower()
    if "zeroshot" in n or "zero_shot" in n:
        return "zeroshot"
    if "finetune" in n:
        return "finetuned"
    return "unknown"


@functools.lru_cache(maxsize=1)
def scan_runs() -> pd.DataFrame:
    """Scan every on-disk run that has BOTH per-episode artifacts.

    Returns one row per run dir: model, dataset, mode, condition, overall_rmse,
    run_path (absolute str), mtime. Single source of truth for selection; never
    reads or writes the stale tracking CSVs.
    """
    rows = []
    for cfgp in sorted(EXP_ROOT.glob("*/*/experiment_config.json")):
        run = cfgp.parent
        # Skip explicitly-excluded / debug runs (dir marked with a leading "_"
        # or containing "excluded", e.g. "_EXCLUDED_single_patient_debug_...").
        if run.name.startswith("_") or "excluded" in run.name.lower():
            continue
        if not (
            (run / "forecasts.npz").exists() and (run / "episodes.parquet").exists()
        ):
            continue
        try:
            cli = json.loads(cfgp.read_text()).get("cli_args", {})
        except Exception:
            cli = {}
        dataset = cli.get("dataset")
        if dataset is None:
            dataset = next((d for d in DATASETS if d in run.name), None)
        rmse = None
        for fn in _SUMMARY_FILES:
            p = run / fn
            if p.exists():
                try:
                    rmse = json.loads(p.read_text()).get("overall_rmse")
                except Exception:
                    rmse = None
                break
        rows.append(
            dict(
                model=run.parent.name,
                dataset=dataset,
                mode=_mode_from_name(run.name),
                condition=condition_bucket(cli.get("covariate_cols")),
                overall_rmse=rmse,
                run_path=str(run),
                mtime=run.stat().st_mtime,
            )
        )
    return pd.DataFrame(rows)


def save_run_index(out_path: str | Path) -> Path:
    """Persist the current run scan to a CSV in the work dir for provenance.

    Writes ONLY inside the rebuttal work dir -- never a tracking table.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scan_runs().to_csv(out_path, index=False)
    return out_path


def _select(
    model: str, dataset: str, condition: str | None = None, mode: str | None = None
) -> pd.DataFrame:
    df = scan_runs()
    m = (df.model == model) & (df.dataset == dataset)
    if condition is not None:
        m &= df.condition == condition
    if mode is not None:
        m &= df["mode"] == mode
    sub = df[m]
    return sub[sub.overall_rmse.notna()].sort_values(
        ["overall_rmse", "mtime"], ascending=[True, False]
    )


def best_run_path(model: str, dataset: str, mode: str | None = None) -> Path | None:
    """Lowest-overall_rmse run for (model, dataset) -- reproduces the paper's
    best-condition cell. Ties / ``_rerun`` collisions broken by newest mtime.

    NOTE: for the ``naive_baseline`` and ``statistical`` *groups* (multiple
    distinct sub-methods per dataset) this returns the single best sub-method;
    use :func:`_select` if you need a specific one.
    """
    sub = _select(model, dataset, mode=mode)
    return Path(sub.iloc[0]["run_path"]) if not sub.empty else None


def run_path_for_condition(
    model: str, dataset: str, condition: str, mode: str | None = None
) -> Path | None:
    """Run for a specific covariate condition (bg_only / iob / iob_cob).

    Used by A4 (bg-only controlled leaderboard) and zero-shot CI selection.
    """
    sub = _select(model, dataset, condition=condition, mode=mode)
    return Path(sub.iloc[0]["run_path"]) if not sub.empty else None


# --------------------------------------------------------------------------
# Metric estimators (operate on a per-episode frame; used by bootstrap)
# --------------------------------------------------------------------------
def pooled_rmse(df: pd.DataFrame) -> float:
    # overall RMSE = sqrt(mean over episodes of per-episode MSE); rmse col is per-episode RMSE
    return float(np.sqrt(np.mean(df["rmse"].to_numpy() ** 2)))


def mean_metric(df: pd.DataFrame, col: str) -> float:
    # Point-only models (e.g. TTM) lack probabilistic columns entirely; return
    # NaN rather than KeyError so the metric is simply absent (matches the paper,
    # where such models have no calibration/WQL row).
    if col not in df.columns:
        return float("nan")
    return float(np.nanmean(df[col].to_numpy()))


def auroc(df: pd.DataFrame, score_col: str = "score_point") -> float:
    from sklearn.metrics import roc_auc_score

    y = df["has_hypo"].to_numpy().astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, df[score_col].to_numpy()))


def auprc(df: pd.DataFrame, score_col: str = "score_point") -> float:
    from sklearn.metrics import average_precision_score

    y = df["has_hypo"].to_numpy().astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, df[score_col].to_numpy()))


# --------------------------------------------------------------------------
# Bootstrap helpers (resample episodes)
# --------------------------------------------------------------------------
def bootstrap_ci(
    df: pd.DataFrame, stat_fn, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05
):
    """Percentile bootstrap CI for a statistic computed on a per-episode frame."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    point = stat_fn(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
        boots[b] = stat_fn(sample)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    stat_fn,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
):
    """Paired bootstrap for stat(A) - stat(B) on the shared episode set.

    Aligns the two frames on ``episode_id`` and resamples the shared episodes
    jointly, so the comparison controls for episode difficulty. Returns
    (delta_point, lo, hi, p_two_sided).
    """
    a = df_a.set_index("episode_id")
    b = df_b.set_index("episode_id")
    shared = a.index.intersection(b.index)
    a = a.loc[shared].reset_index()
    b = b.loc[shared].reset_index()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(shared))
    delta_point = stat_fn(a) - stat_fn(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = stat_fn(a.iloc[take]) - stat_fn(b.iloc[take])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value (proportion crossing zero, doubled)
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return (
        float(delta_point),
        float(lo),
        float(hi),
        float(min(p, 1.0)),
        int(len(shared)),
    )


# --------------------------------------------------------------------------
# PATIENT-CLUSTER bootstrap (episodes are nested within patients)
# --------------------------------------------------------------------------
# Episodes are NOT i.i.d.: each patient contributes many midnight episodes
# (median 4-13, up to ~320). Episode-level resampling therefore UNDERSTATES
# uncertainty (effective N ~ #patients, not #episodes). The cluster bootstrap
# resamples PATIENTS with replacement and takes all of a patient's episodes,
# preserving within-patient correlation -> honest (wider) CIs.


def _cluster_index(df: pd.DataFrame, cluster_col: str = "pid") -> list[np.ndarray]:
    d = df.reset_index(drop=True)
    return [g.to_numpy() for _, g in d.groupby(cluster_col).groups.items()]


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn,
    cluster_col: str = "pid",
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
):
    """Percentile CI resampling PATIENTS (clusters) with replacement."""
    d = df.reset_index(drop=True)
    # positional row-index array per cluster (patient)
    groups = [np.asarray(v) for v in d.groupby(cluster_col).indices.values()]
    rng = np.random.default_rng(seed)
    k = len(groups)
    point = stat_fn(d)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.integers(0, k, size=k)
        rows = np.concatenate([groups[i] for i in take])
        boots[b] = stat_fn(d.iloc[rows])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_cluster_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    stat_fn,
    cluster_col: str = "pid",
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
):
    """Paired PATIENT-cluster bootstrap for stat(A)-stat(B) on shared episodes.

    Resamples the shared patients jointly (same patients drawn for A and B),
    controlling for both patient and episode difficulty. Returns
    (delta_point, lo, hi, p_two_sided, n_clusters).
    """
    a = df_a.set_index("episode_id")
    b = df_b.set_index("episode_id")
    shared = a.index.intersection(b.index)
    a = a.loc[shared].reset_index(drop=True)
    b = b.loc[shared].reset_index(drop=True)
    # cluster positions (a and b are row-aligned on the shared episode set)
    ga = a.groupby(cluster_col).indices
    keys = list(ga.keys())
    groups = [np.asarray(ga[k]) for k in keys]
    rng = np.random.default_rng(seed)
    k = len(groups)
    delta_point = stat_fn(a) - stat_fn(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, k, size=k)
        rows = np.concatenate([groups[j] for j in take])
        boots[i] = stat_fn(a.iloc[rows]) - stat_fn(b.iloc[rows])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(delta_point), float(lo), float(hi), float(min(p, 1.0)), int(k)
