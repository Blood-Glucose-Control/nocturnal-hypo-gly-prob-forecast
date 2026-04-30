# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Grand summary aggregator for nocturnal forecasting evaluations.

Builds a wide results table (super-columns = dataset, sub-columns = metric,
row sections = model class) plus a model-property matrix and a long-form
companion table of best runs.

Inputs:
    One or more `summary.csv` files produced by NocturnalSummarizer.
Outputs (returned DataFrames; the CLI script writes them to disk):
    * model_properties_df  -- architecture-level capabilities per model family.
    * results_wide_df      -- best run per (model, dataset, cov_bucket).
    * best_runs_long_df    -- tidy version of the wide table.
    * missing_combos_df    -- (model, dataset, cov_bucket) cells that should
      exist (per MODEL_PROPERTIES) but have no run.

Module: experiments.nocturnal.grand_summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS: tuple[str, ...] = (
    "aleppo_2017",
    "brown_2019",
    "lynch_2022",
    "tamborlane_2008",
)

DEFAULT_METRICS: tuple[str, ...] = (
    "rmse",
    "wql",
    "sharpness_50",
    "coverage_50",
    "sharpness_80",
    "coverage_80",
    "brier_3_9",
    "dilate_g001",
    "shape_g001",
    "temporal_g001",
)

# Buckets for the "best version of each model" rows in the wide table.
COV_BUCKETS: tuple[str, ...] = ("zero_shot", "bg_only", "iob", "cob", "iob_cob")

COV_BUCKET_LABELS: dict[str, str] = {
    "zero_shot": "Best Zero-Shot",
    "bg_only": "Best BG-only",
    "iob": "Best +IOB",
    "cob": "Best +COB",
    "iob_cob": "Best +IOB+COB",
}

# Dataset-specific covariate availability.
# Each dataset only has a subset of clinical covariates available; only the
# corresponding bucket can be legitimately populated for that dataset.
# Runs in any other covariate bucket are flagged as MISPLACED.
VALID_COVARIATE_BUCKETS_BY_DATASET: dict[str, frozenset[str]] = {
    "aleppo_2017": frozenset({"iob_cob"}),  # both IOB and COB available
    "brown_2019": frozenset({"iob"}),  # IOB only
    "lynch_2022": frozenset({"iob"}),  # IOB only
    "tamborlane_2008": frozenset(),  # BG-only; no clinical covariates available
}

# Model class taxonomy. Order here drives row order in the wide table.
MODEL_CLASS_ORDER: tuple[str, ...] = (
    "naive",
    "statistical",
    "deep_learning",
    "foundation",
)

MODEL_CLASS_LABELS: dict[str, str] = {
    "naive": "Naive baselines",
    "statistical": "Statistical",
    "deep_learning": "Deep learning",
    "foundation": "Foundation (pre-trained)",
}

# Architecture-level capabilities per model family.
# Keys match the `model` column emitted by NocturnalSummarizer (multi-variant
# models keep their `parent/variant` form, e.g. `statistical/AutoARIMA`).
#
# Booleans:
#   zero_shot_capable       -- pretrained weights usable with no fine-tune?
#   fine_tunable            -- supports per-dataset fine-tune in our pipeline?
#   supports_past_covariates-- accepts past covariates (IOB / COB / IA)?
#   univariate_only         -- forced to single channel?
#   probabilistic           -- emits quantiles / sample paths?
MODEL_PROPERTIES: dict[str, dict[str, Any]] = {
    # ---- Naive baselines ----
    # Naive baselines are deterministic; we evaluate them through the same
    # pipeline as fine-tuned models (mode=finetuned) so we mark them
    # fine_tunable rather than zero_shot_capable to match how the runs are
    # recorded.  Conceptually they have no learnable parameters either way.
    "naive_baseline/Naive": {
        "class": "naive",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Last-observation-carried-forward. Deterministic; no learnable parameters."
        ),
    },
    "naive_baseline/Average": {
        "class": "naive",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Mean of the context window. Deterministic; no learnable parameters."
        ),
    },
    # ---- Statistical ----
    "statistical/AutoARIMA": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Box-Jenkins ARIMA with automated (p,d,q) order selection per series; "
            "Gaussian predictive distribution."
        ),
    },
    "statistical/Theta": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Theta decomposition: drift line + simple exponential smoothing of "
            "deseasonalised series. Point forecasts only."
        ),
    },
    "statistical/NPTS": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Non-Parametric Time Series: empirical resampling of historical values "
            "with kernel-weighted recency."
        ),
    },
    # ---- Deep learning (trained from scratch / on our data) ----
    "deepar": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Autoregressive LSTM emitting parametric (Student-t/Gaussian) "
            "likelihood per step. AutoGluon only exposes known (future-available) "
            "covariates for DeepAR; past-only covariates like IOB/COB cannot be "
            "used without data leakage."
        ),
    },
    "tft": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Temporal Fusion Transformer: variable-selection networks + LSTM "
            "encoder/decoder + multi-head interpretable attention + quantile head."
        ),
    },
    "patchtst": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Patch-tokenised transformer encoder, channel-independent. "
            "Point/quantile forecasts; no native covariate support."
        ),
    },
    "tide": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "Time-series Dense Encoder: pure MLP encoder-decoder with residual "
            "connections; no attention. Point forecasts. AutoGluon only exposes "
            "known (future-available) covariates for TiDE; past-only covariates "
            "like IOB/COB cannot be used without data leakage."
        ),
    },
    "timegrad": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Autoregressive denoising-diffusion RNN: LSTM hidden state conditions "
            "a per-step DDPM that samples next observation. Uses the pts library "
            "directly (not AutoGluon); only BG context is consumed — no covariate "
            "inputs are wired in."
        ),
    },
    # ---- Foundation (pre-trained) ----
    "chronos2": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "T5 encoder-decoder over quantised value tokens (Chronos-2). "
            "Native past + static covariates; quantile sampling."
        ),
    },
    "moirai": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Masked encoder transformer with patch embeddings and multi-patch-size "
            "attention; any-variate forecasting."
        ),
    },
    "toto": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Datadog Toto: decoder-only transformer foundation model with "
            "proprietary tokeniser, trained on observability time-series."
        ),
    },
    "timesfm": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Google TimesFM: decoder-only transformer with input/output patching; "
            "univariate; quantile head."
        ),
    },
    "ttm": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "TinyTimeMixer (IBM): mixer-style channel/patch MLPs (no attention); "
            "very small (~1M params). Point forecasts."
        ),
    },
    "sundial": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": False,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "THUML Sundial: generative diffusion-based transformer foundation model "
            "(ICML 2025). Draws sample paths via autoregressive generation; "
            "quantiles derived from samples. Used zero-shot only in this study."
        ),
    },
    "moment": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "MOMENT (CMU): masked-reconstruction transformer foundation model "
            "with patch embeddings; channel-independent variant supports past "
            "covariates via concatenation. Point forecasts."
        ),
    },
}


def model_supports_past_covariates(model_name: str) -> bool:
    """Return True if *model_name* can consume past covariate columns.

    Looks up ``MODEL_PROPERTIES`` by exact key first, then by the base model
    name (stripping any ``/variant`` suffix).  Returns ``False`` for unknown
    models so that unrecognised names are treated conservatively.
    """
    if model_name in MODEL_PROPERTIES:
        return MODEL_PROPERTIES[model_name]["supports_past_covariates"]
    base = model_name.split("/")[0]
    if base in MODEL_PROPERTIES:
        return MODEL_PROPERTIES[base]["supports_past_covariates"]
    return False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_summary(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load + concat one or more summary.csv files, recording their source."""
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            log.warning("Summary file not found: %s", path)
            continue
        df = pd.read_csv(path)
        df["_source_summary"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"None of the provided summary files exist: {paths}")
    return pd.concat(frames, ignore_index=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate evals on (model, dataset, mode, checkpoint, ctx_fh).

    Keeps the row with the latest timestamp.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("timestamp")
    keys = ["model", "dataset", "mode", "checkpoint", "ctx_fh"]
    # `checkpoint` is NaN for zero-shot; fillna so groupby keeps zero-shots
    # collapsed too.
    for k in keys:
        if k in out.columns:
            out[k] = out[k].fillna("")
    out = out.drop_duplicates(subset=keys, keep="last")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Covariate-bucket attachment
# ---------------------------------------------------------------------------


def read_covariate_cols(run_path: str | Path) -> list[str] | None:
    """Read `cli_args.covariate_cols` from a run's experiment_config.json."""
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
    cols = cli.get("covariate_cols")
    if cols is None:
        # Fall back to model_config.covariate_cols (populated when covariates
        # are specified in the model YAML rather than as CLI args).
        model_cfg = cfg.get("model_config", {}) or {}
        cols = model_cfg.get("covariate_cols")
    if cols is None:
        return []
    if isinstance(cols, list):
        return [str(c) for c in cols]
    return []


# Carb-related covariates that count as the COB-family signal.
_CARB_COVARIATES: frozenset[str] = frozenset({"cob", "carb_availability", "carbs"})
# Insulin-related (beyond plain IOB) covariates that count as IOB-family signal.
_INSULIN_COVARIATES: frozenset[str] = frozenset({"iob", "ia", "insulin_availability"})


def bucket_from_covariates(mode: str, cov_cols: Iterable[str] | None) -> str:
    """Map (mode, covariate_cols) to one of COV_BUCKETS.

    ``mode`` should be the normalised string produced by the summarizer
    (hyphens stripped, lower-cased), e.g. ``"zeroshot"`` or ``"finetuned"``.

    Buckets:
        zero_shot -- model run in zero-shot mode
        iob_cob   -- both insulin and carb signals present (e.g. aleppo_2017)
        iob       -- insulin signal only (e.g. brown_2019, lynch_2022)
        cob       -- carb signal only (e.g. tamborlane_2008)
        bg_only   -- no non-BG covariates
    """
    if mode == "zeroshot":
        return "zero_shot"
    cset = {c.lower() for c in (cov_cols or [])}
    # Drop time-feature names that aren't clinical BG covariates
    cset -= {"hour_sin", "hour_cos", "minute_sin", "minute_cos", "day_sin", "day_cos"}
    if not cset:
        return "bg_only"
    has_carb = bool(cset & _CARB_COVARIATES)
    has_insulin = bool(cset & _INSULIN_COVARIATES)
    if has_carb and has_insulin:
        return "iob_cob"
    if has_carb:
        return "cob"
    if has_insulin:
        return "iob"
    # Unknown covariate set: treat as BG-only and warn
    log.warning("Unrecognised covariate set %s — bucketing as bg_only", sorted(cset))
    return "bg_only"


def attach_covariate_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add `cov_bucket` and `cov_variant` columns to a summary DataFrame.

    Reads each run's `experiment_config.json` (cached in-process by run_path).
    """
    cache: dict[str, list[str] | None] = {}
    buckets: list[str] = []
    variants: list[str] = []
    for _, row in df.iterrows():
        run_path = str(row.get("run_path", ""))
        mode = str(row.get("mode", ""))
        if run_path not in cache:
            cache[run_path] = read_covariate_cols(run_path)
        cov_cols = cache[run_path]
        buckets.append(bucket_from_covariates(mode, cov_cols))
        variants.append(",".join(sorted(cov_cols)) if cov_cols else "")
    out = df.copy()
    out["cov_bucket"] = buckets
    out["cov_variant"] = variants
    return out


# ---------------------------------------------------------------------------
# Best-run selection + pivoting
# ---------------------------------------------------------------------------


def pick_best(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("model", "dataset", "cov_bucket"),
) -> pd.DataFrame:
    """Pick one row per group: lowest RMSE, tie-break on WQL then latest ts."""
    out = df.copy()
    out["rmse"] = pd.to_numeric(out["rmse"], errors="coerce")
    out["wql"] = pd.to_numeric(out["wql"], errors="coerce")
    out = out.dropna(subset=["rmse"])
    # Stable sort: latest timestamp first, then wql, then rmse, so groupby.first
    # yields lowest rmse, lowest wql, latest ts within ties.
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(
        ["rmse", "wql", "timestamp"],
        ascending=[True, True, False],
        na_position="last",
    )
    return out.groupby(list(group_cols), as_index=False).first()


def _model_class(model: str) -> str:
    props = MODEL_PROPERTIES.get(model)
    if props:
        return props["class"]
    # Fallback: try parent (e.g. statistical/Foo -> statistical)
    parent = model.split("/", 1)[0]
    parent_props = MODEL_PROPERTIES.get(parent)
    if parent_props:
        return parent_props["class"]
    return "deep_learning"


def _model_sort_key(model: str) -> tuple[int, str]:
    cls = _model_class(model)
    cls_idx = MODEL_CLASS_ORDER.index(cls) if cls in MODEL_CLASS_ORDER else 99
    return (cls_idx, model)


def pivot_wide(
    df_best: pd.DataFrame,
    datasets: Sequence[str] = DATASETS,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Pivot best-runs DF to wide form.

    Index : MultiIndex (model_class, model, cov_bucket)
    Cols  : MultiIndex (dataset, metric)
    """
    rows: dict[tuple[str, str, str], dict[tuple[str, str], float]] = {}
    for _, r in df_best.iterrows():
        model = str(r["model"])
        cls = _model_class(model)
        bucket = str(r["cov_bucket"])
        ds = str(r["dataset"])
        if ds not in datasets or bucket not in COV_BUCKETS:
            continue
        key = (cls, model, bucket)
        rows.setdefault(key, {})
        for m in metrics:
            val = r.get(m)
            try:
                rows[key][(ds, m)] = float(val) if pd.notna(val) else float("nan")
            except (TypeError, ValueError):
                rows[key][(ds, m)] = float("nan")

    if not rows:
        return pd.DataFrame()

    col_index = pd.MultiIndex.from_product(
        [list(datasets), list(metrics)], names=["dataset", "metric"]
    )
    sorted_keys = sorted(
        rows.keys(),
        key=lambda k: (
            MODEL_CLASS_ORDER.index(k[0]) if k[0] in MODEL_CLASS_ORDER else 99,
            k[1],  # model
            COV_BUCKETS.index(k[2]) if k[2] in COV_BUCKETS else 99,
        ),
    )
    row_index = pd.MultiIndex.from_tuples(
        sorted_keys, names=["model_class", "model", "cov_bucket"]
    )
    data = [[rows[k].get(c, float("nan")) for c in col_index] for k in sorted_keys]
    return pd.DataFrame(data, index=row_index, columns=col_index)


# ---------------------------------------------------------------------------
# Model property matrix + missing-combinations
# ---------------------------------------------------------------------------


def build_model_properties_df() -> pd.DataFrame:
    rows = []
    for model, props in MODEL_PROPERTIES.items():
        rows.append({"model": model, **props})
    df = pd.DataFrame(rows)
    df["_class_idx"] = df["class"].map(
        lambda c: MODEL_CLASS_ORDER.index(c) if c in MODEL_CLASS_ORDER else 99
    )
    df = df.sort_values(["_class_idx", "model"]).drop(columns=["_class_idx"])
    return df.reset_index(drop=True)


def expected_combinations() -> set[tuple[str, str, str]]:
    """Set of (model, dataset, cov_bucket) that *should* exist per properties.

    Dataset-aware: covariate buckets beyond bg_only are only expected where the
    dataset actually has those covariates (see VALID_COVARIATE_BUCKETS_BY_DATASET).
    """
    expected: set[tuple[str, str, str]] = set()
    for model, props in MODEL_PROPERTIES.items():
        base_buckets: list[str] = []
        if props["zero_shot_capable"]:
            base_buckets.append("zero_shot")
        if props["fine_tunable"]:
            base_buckets.append("bg_only")
        for ds in DATASETS:
            for b in base_buckets:
                expected.add((model, ds, b))
            if props["fine_tunable"] and props["supports_past_covariates"]:
                valid_cov = VALID_COVARIATE_BUCKETS_BY_DATASET.get(ds, frozenset())
                for b in valid_cov:
                    expected.add((model, ds, b))
    return expected


def find_missing(df_best: pd.DataFrame) -> pd.DataFrame:
    have = set(
        zip(
            df_best["model"].astype(str),
            df_best["dataset"].astype(str),
            df_best["cov_bucket"].astype(str),
        )
    )
    missing = expected_combinations() - have
    rows = [
        {
            "model": m,
            "dataset": d,
            "cov_bucket": b,
            "model_class": _model_class(m),
        }
        for (m, d, b) in sorted(
            missing, key=lambda x: (_model_sort_key(x[0]), x[1], x[2])
        )
    ]
    return pd.DataFrame(rows)


def find_misplaced(df_best: pd.DataFrame) -> pd.DataFrame:
    """Find runs recorded in a covariate bucket that is invalid for their dataset.

    A run is *misplaced* when its cov_bucket is not in {zero_shot, bg_only} AND
    not in VALID_COVARIATE_BUCKETS_BY_DATASET for the run's dataset.  These are
    genuine errors: either the pipeline was launched with the wrong covariate
    list, or the bucketing logic needs a fix.
    """
    rows = []
    universal_buckets = frozenset({"zero_shot", "bg_only"})
    for _, r in df_best.iterrows():
        bucket = str(r["cov_bucket"])
        ds = str(r["dataset"])
        if bucket in universal_buckets:
            continue
        valid = VALID_COVARIATE_BUCKETS_BY_DATASET.get(ds, frozenset())
        if bucket not in valid:
            rows.append(
                {
                    "model": r["model"],
                    "dataset": ds,
                    "cov_bucket": bucket,
                    "cov_variant": r.get("cov_variant", ""),
                    "model_class": _model_class(str(r["model"])),
                    "run_path": r.get("run_path", ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "cov_bucket",
                "cov_variant",
                "model_class",
                "run_path",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model_class", "model", "dataset", "cov_bucket"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def build_grand_summary(
    summary_paths: Sequence[str | Path],
    datasets: Sequence[str] = DATASETS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ctx_filter: int | None = None,
    forecast_filter: int | None = 96,
) -> dict[str, pd.DataFrame]:
    """End-to-end pipeline. Returns a dict of named DataFrames."""
    raw = load_summary(summary_paths)
    if forecast_filter is not None and "forecast_length" in raw.columns:
        raw = raw[raw["forecast_length"] == forecast_filter]
    if ctx_filter is not None and "context_length" in raw.columns:
        raw = raw[raw["context_length"] == ctx_filter]
    # Only include models registered in MODEL_PROPERTIES; unrecognised names
    # indicate deprecated or misconfigured runs (e.g. AutoARIMA+IOB which
    # silently ignores IOB due to AutoGluon's known_covariates_names limitation).
    raw = raw[raw["model"].isin(MODEL_PROPERTIES)]
    deduped = dedupe(raw)
    bucketed = attach_covariate_bucket(deduped)
    best = pick_best(bucketed)
    wide = pivot_wide(best, datasets=datasets, metrics=metrics)
    missing = find_missing(best)
    misplaced = find_misplaced(best)
    properties = build_model_properties_df()
    return {
        "results_wide": wide,
        "best_runs_long": best,
        "missing_combinations": missing,
        "misplaced_combinations": misplaced,
        "model_properties": properties,
        "all_runs_bucketed": bucketed,
    }
