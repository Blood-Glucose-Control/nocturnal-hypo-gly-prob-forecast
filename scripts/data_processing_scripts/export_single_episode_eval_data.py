#!/usr/bin/env python3
"""Export single-episode evaluation data for visualization.

This script:
1) Selects best runs from summary.csv for brown_2019 at 96fh,
   separately for zero-shot and fine-tuned per model family.
2) Extracts one midnight episode (context + horizon) with all available
   columns (target and covariates) for a chosen patient/episode id.
3) Loads each selected model/run and generates forecasts for that same episode.
4) Writes one shared base CSV and separate per-model CSV files.

Example:
    python scripts/visualization/export_single_episode_eval_data.py \
        --episode-id bro_14::ep060
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.utils import get_patient_column
from src.data.versioning.dataset_registry import DatasetRegistry
from src.evaluation.episode_builders import build_midnight_episodes
from src.models import create_model_and_config

LOGGER = logging.getLogger(__name__)
INTERVAL_MINS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-model single-episode data for visualization"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="experiments/nocturnal_forecasting/summary.csv",
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="brown_2019",
        help="Dataset name to filter summary rows",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=96,
        help="Forecast horizon to filter summary rows",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length to use for episode extraction",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        required=True,
        help="Episode id in format <patient_id>::epNNN (e.g., bro_14::ep060)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_10pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/nocturnal_forecasting/episode_visualization_data",
        help="Output root directory",
    )
    return parser.parse_args()


def parse_episode_id(episode_id: str) -> tuple[str, int]:
    if "::ep" not in episode_id:
        raise ValueError(
            f"Invalid episode id '{episode_id}'. Expected format <patient_id>::epNNN"
        )
    patient_id, ep_suffix = episode_id.split("::ep", maxsplit=1)
    if not ep_suffix.isdigit():
        raise ValueError(
            f"Invalid episode id '{episode_id}'. Episode suffix must be numeric"
        )
    return patient_id, int(ep_suffix)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    if "datetime" not in out.columns:
        raise ValueError("Holdout data must have a DatetimeIndex or 'datetime' column")

    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.set_index("datetime").sort_index()
    return out


def choose_best_rows(
    summary_df: pd.DataFrame,
    dataset: str,
    forecast_length: int,
) -> pd.DataFrame:
    filtered = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["forecast_length"] == forecast_length)
    ].copy()

    filtered = filtered.dropna(subset=["rmse"])
    filtered = filtered[filtered["mode"].isin(["zeroshot", "finetuned"])]

    if filtered.empty:
        raise ValueError(
            f"No rows found for dataset={dataset}, forecast_length={forecast_length}"
        )

    best = (
        filtered.sort_values("rmse")
        .groupby(["model", "mode"], as_index=False)
        .first()
        .sort_values(["model", "mode"])
    )
    return best


def load_experiment_config(run_path: Path) -> dict[str, Any]:
    cfg_path = run_path / "experiment_config.json"
    if not cfg_path.exists():
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint(run_row: pd.Series, exp_config: dict[str, Any]) -> str | None:
    mode = str(run_row["mode"])
    run_path = Path(str(run_row["run_path"]))

    if mode == "zeroshot":
        return None

    cli_args = exp_config.get("cli_args", {}) if isinstance(exp_config, dict) else {}
    checkpoint = cli_args.get("checkpoint")
    if checkpoint:
        return str(checkpoint)

    # Fallbacks for finetuned runs when checkpoint isn't recorded.
    fallback_candidates = [
        run_path / "model.pt",
        run_path,
    ]
    for cand in fallback_candidates:
        if cand.exists():
            return str(cand)

    return None


def infer_covariates_from_config(config: Any) -> list[str]:
    covs: list[str] = []

    if hasattr(config, "covariate_cols") and getattr(config, "covariate_cols"):
        covs.extend(list(getattr(config, "covariate_cols")))

    if hasattr(config, "is_multitarget") and getattr(config, "is_multitarget"):
        for col in getattr(config, "joint_target_cols", []) or []:
            if col != "bg_mM":
                covs.append(col)

    deduped = []
    seen = set()
    for c in covs:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def _normalize_anchor(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def try_load_saved_episode_prediction(
    run_path: Path,
    patient_id: str,
    anchor: pd.Timestamp,
    forecast_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pred, target) from nocturnal_results.json when available."""
    results_path = run_path / "nocturnal_results.json"
    if not results_path.exists():
        return None

    try:
        with results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    per_episode = payload.get("per_episode", [])
    if not isinstance(per_episode, list):
        return None

    anchor_norm = pd.Timestamp(anchor)
    matched = None
    for ep in per_episode:
        if not isinstance(ep, dict):
            continue
        if str(ep.get("patient_id")) != patient_id:
            continue
        ep_anchor = _normalize_anchor(ep.get("anchor"))
        if ep_anchor is None:
            continue
        if ep_anchor == anchor_norm:
            matched = ep
            break

    if not matched:
        return None

    pred = np.asarray(matched.get("pred", []), dtype=float)
    tgt = np.asarray(matched.get("target_bg", []), dtype=float)
    if len(pred) == 0 or len(tgt) == 0:
        return None

    n = min(forecast_length, len(pred), len(tgt))
    return pred[:n], tgt[:n]


def build_episode_base_df(
    patient_df: pd.DataFrame,
    anchor: pd.Timestamp,
    context_length: int,
    forecast_length: int,
) -> pd.DataFrame:
    dt = pd.Timedelta(minutes=INTERVAL_MINS)
    start = anchor - context_length * dt
    end = anchor + forecast_length * dt
    index = pd.date_range(
        start=start, end=end, freq=f"{INTERVAL_MINS}min", inclusive="left"
    )

    window_df = patient_df.reindex(index).copy()
    window_df = window_df.reset_index(names="timestamp")

    total = len(window_df)
    split = np.where(np.arange(total) < context_length, "context", "forecast")
    step = np.arange(total) - context_length

    window_df.insert(1, "segment", split)
    window_df.insert(2, "step", step)
    window_df.insert(3, "hours_from_anchor", step * INTERVAL_MINS / 60.0)
    return window_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    patient_id, ep_index = parse_episode_id(args.episode_id)

    summary_df = pd.read_csv(summary_path)
    best_runs = choose_best_rows(summary_df, args.dataset, args.forecast_length)
    LOGGER.info("Selected %d best model/mode rows", len(best_runs))

    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_df = registry.load_holdout_data_only(args.dataset)
    patient_col = get_patient_column(holdout_df)

    patient_df = holdout_df[holdout_df[patient_col] == patient_id].copy()
    if patient_df.empty:
        raise ValueError(f"Patient '{patient_id}' not found in holdout data")
    patient_df = ensure_datetime_index(patient_df)

    episodes_for_index, _ = build_midnight_episodes(
        patient_df,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        target_col="bg_mM",
        covariate_cols=None,
        interval_mins=INTERVAL_MINS,
    )
    if ep_index >= len(episodes_for_index):
        raise ValueError(
            f"Episode index {ep_index} out of range for patient {patient_id}. "
            f"Available episodes: 0..{max(0, len(episodes_for_index)-1)}"
        )

    selected_episode = episodes_for_index[ep_index]
    anchor = pd.Timestamp(selected_episode["anchor"])
    target_bg = np.asarray(selected_episode["target_bg"])

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{patient_id}_ep{ep_index:03d}_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = build_episode_base_df(
        patient_df=patient_df,
        anchor=anchor,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
    )
    base_path = out_dir / "episode_base.csv"
    base_df.to_csv(base_path, index=False)

    metadata: dict[str, Any] = {
        "dataset": args.dataset,
        "episode_id": args.episode_id,
        "patient_id": patient_id,
        "episode_index": ep_index,
        "anchor": anchor.isoformat(),
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
        "summary_csv": str(summary_path),
        "selected_runs": [],
        "skipped_runs": [],
    }

    combined_rows: list[pd.DataFrame] = []

    for _, run_row in best_runs.iterrows():
        model_name = str(run_row["model"])
        mode = str(run_row["mode"])
        run_id = str(run_row["run_id"])
        run_path = Path(str(run_row["run_path"]))

        LOGGER.info("Running %s (%s) from %s", model_name, mode, run_id)

        saved = try_load_saved_episode_prediction(
            run_path=run_path,
            patient_id=patient_id,
            anchor=anchor,
            forecast_length=args.forecast_length,
        )

        prediction_source = "live_inference"
        covariates: list[str] = []

        if saved is not None:
            pred, tgt = saved
            rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
            prediction_source = "saved_results"
        else:
            exp_config = load_experiment_config(run_path)
            checkpoint = resolve_checkpoint(run_row, exp_config)

            cli_args = (
                exp_config.get("cli_args", {}) if isinstance(exp_config, dict) else {}
            )
            model_cfg = (
                exp_config.get("model_config", {})
                if isinstance(exp_config, dict)
                else {}
            )

            model_kwargs = dict(model_cfg) if isinstance(model_cfg, dict) else {}
            model_kwargs["context_length"] = int(
                run_row.get("context_length", args.context_length)
            )
            model_kwargs["forecast_length"] = int(
                run_row.get("forecast_length", args.forecast_length)
            )

            # Keep explicit covariates from the original run when available.
            if "covariate_cols" in cli_args and cli_args["covariate_cols"]:
                model_kwargs["covariate_cols"] = cli_args["covariate_cols"]

            try:
                model, config = create_model_and_config(
                    model_name,
                    checkpoint=checkpoint,
                    **model_kwargs,
                )
            except Exception as exc:
                LOGGER.exception("Failed to initialize model %s (%s)", model_name, mode)
                metadata["skipped_runs"].append(
                    {
                        "model": model_name,
                        "mode": mode,
                        "run_id": run_id,
                        "reason": f"init_failed: {exc}",
                    }
                )
                continue

            covariates = infer_covariates_from_config(config)

            eps_for_model, _ = build_midnight_episodes(
                patient_df,
                context_length=model_kwargs["context_length"],
                forecast_length=model_kwargs["forecast_length"],
                target_col="bg_mM",
                covariate_cols=covariates or None,
                interval_mins=INTERVAL_MINS,
            )

            if ep_index >= len(eps_for_model):
                metadata["skipped_runs"].append(
                    {
                        "model": model_name,
                        "mode": mode,
                        "run_id": run_id,
                        "reason": "episode_not_available_for_model_context_horizon",
                    }
                )
                continue

            model_episode = eps_for_model[ep_index]
            context_df = (
                model_episode["context_df"].copy().reset_index(names="datetime")
            )
            context_df["p_num"] = patient_id
            context_df["episode_id"] = args.episode_id

            try:
                batch_pred = model.predict_batch(context_df, episode_col="episode_id")
                pred = np.asarray(batch_pred[args.episode_id])
            except Exception as exc:
                LOGGER.exception(
                    "Prediction failed for model %s (%s)", model_name, mode
                )
                metadata["skipped_runs"].append(
                    {
                        "model": model_name,
                        "mode": mode,
                        "run_id": run_id,
                        "reason": f"predict_failed: {exc}",
                    }
                )
                continue

            pred = pred[: args.forecast_length]
            tgt = target_bg[: len(pred)]
            rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))

            exp_config = load_experiment_config(run_path)
            checkpoint = resolve_checkpoint(run_row, exp_config)

        if saved is not None:
            checkpoint = None

        forecast_slice = (
            base_df[base_df["segment"] == "forecast"].copy().iloc[: len(pred)]
        )
        forecast_slice = forecast_slice[
            ["timestamp", "segment", "step", "hours_from_anchor", "bg_mM"]
        ].copy()
        forecast_slice = forecast_slice.rename(columns={"bg_mM": "target_bg_mM"})
        forecast_slice["pred_bg_mM"] = pred
        forecast_slice["model"] = model_name
        forecast_slice["mode"] = mode
        forecast_slice["run_id"] = run_id
        forecast_slice["run_path"] = str(run_path)
        forecast_slice["source_rmse_summary"] = float(run_row["rmse"])
        forecast_slice["episode_rmse"] = rmse

        model_file = out_dir / f"forecast_{model_name}_{mode}.csv"
        forecast_slice.to_csv(model_file, index=False)
        combined_rows.append(forecast_slice)

        metadata["selected_runs"].append(
            {
                "model": model_name,
                "mode": mode,
                "run_id": run_id,
                "run_path": str(run_path),
                "checkpoint": checkpoint,
                "prediction_source": prediction_source,
                "model_covariates": covariates,
                "summary_rmse": float(run_row["rmse"]),
                "episode_rmse": rmse,
                "output_file": str(model_file),
            }
        )

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(out_dir / "forecast_all_models.csv", index=False)

    metadata_path = out_dir / "export_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Wrote base data: %s", base_path)
    LOGGER.info("Wrote metadata: %s", metadata_path)
    LOGGER.info("Completed. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
