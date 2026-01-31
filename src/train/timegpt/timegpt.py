# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher@gluroo.com/cjrisi@uwaterloo.ca

"""
TimeGPT finetuning pipeline using the Nixtla SDK.

This mirrors the TTM finetuning flow:
- YAML-driven configuration
- Optional run directory for saving configs, forecasts, and metrics
- Uses cached, processed patient CSVs to avoid recomputation
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from nixtla import NixtlaClient

# Ensure project root is on sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.cache_manager import get_cache_manager  # noqa: E402
from src.utils.time_series_helper import get_interval_minutes  # noqa: E402


def info_print(*args, **kwargs):
    """Log to stderr so messages show up in Slurm error streams."""
    print("INFO:", *args, file=sys.stderr, **kwargs, flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TimeGPT finetuning with YAML configuration"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Directory to save run outputs and logs",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _load_processed_series(
    dataset_name: str,
    target_col: str,
    time_col: str,
    id_col: str,
    resolution_min: Optional[int] = None,
    freq: Optional[str] = None,
    min_length: int = 1,
) -> pd.DataFrame:
    """
    Load processed CSVs from cache and return a long dataframe with
    [id_col, time_col, target_col].
    """
    cache_manager = get_cache_manager()
    cache_dir = cache_manager.get_absolute_path_by_type(dataset_name, "processed")
    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {cache_dir}")

    info_print(f"Found {len(csv_files)} processed CSV files in {cache_dir}")
    series_frames = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        patient_id = filename.replace(".csv", "")

        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        except Exception as exc:
            info_print(f"Skipping {csv_file} due to load error: {exc}")
            continue

        if resolution_min is not None and get_interval_minutes(df) != resolution_min:
            continue

        df = df.reset_index().rename(columns={"index": time_col})
        if time_col not in df.columns:
            raise ValueError(
                f"Time column '{time_col}' not found after resetting index for {csv_file}"
            )

        if target_col not in df.columns:
            info_print(
                f"Target column '{target_col}' not in {csv_file}; available: {df.columns}"
            )
            continue

        # Normalize timestamps to the target frequency boundary
        effective_freq = freq or (f"{resolution_min}min" if resolution_min else None)
        df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize(None)
        if effective_freq:
            df[time_col] = df[time_col].dt.floor(effective_freq)

        subset = df[[time_col, target_col]].dropna()
        if effective_freq:
            # Collapse any duplicates within the same time bucket by averaging
            subset = subset.groupby(time_col, as_index=False)[target_col].mean()
        subset = subset.assign(**{id_col: patient_id})

        # Enforce regular frequency and remove duplicates for Nixtla
        subset = (
            subset.sort_values(time_col)
            .drop_duplicates(subset=[time_col])
            .drop_duplicates(subset=[time_col, id_col])
        )
        if effective_freq:
            subset = (
                subset.set_index(time_col)[target_col].resample(effective_freq).mean()
            )
            subset = subset.to_frame(name=target_col)
            subset[target_col] = subset[target_col].interpolate(limit_direction="both")
            subset = subset.reset_index()
            subset[time_col] = pd.to_datetime(subset[time_col])
            subset[id_col] = patient_id

        # Drop any remaining missing values and skip too-short series
        subset = subset.dropna(subset=[target_col, time_col, id_col])

        if len(subset) < min_length:
            info_print(
                f"Skipping patient {patient_id}: insufficient rows after regularization"
            )
            continue

        series_frames.append(subset)

    if not series_frames:
        raise ValueError("No usable patient series found with the requested settings")

    combined = pd.concat(series_frames).rename(
        columns={time_col: time_col, target_col: target_col, id_col: id_col}
    )
    info_print(
        f"Loaded series for {combined[id_col].nunique()} patients with "
        f"{len(combined)} total rows"
    )
    # Final sort and clean across all series
    combined[time_col] = pd.to_datetime(combined[time_col]).dt.tz_localize(None)
    combined = combined.sort_values([id_col, time_col]).drop_duplicates(
        subset=[id_col, time_col]
    )
    return combined


def _split_train_test(
    df: pd.DataFrame, id_col: str, time_col: str, horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split per-series into train (all but last h) and test (last h)."""
    train_frames = []
    test_frames = []

    for uid, group in df.groupby(id_col):
        ordered = group.sort_values(time_col)
        if len(ordered) <= horizon:
            info_print(
                f"Series {uid} too short for horizon {horizon}; using all rows for training"
            )
            train_frames.append(ordered)
            continue

        train_frames.append(ordered.iloc[:-horizon])
        test_frames.append(ordered.iloc[-horizon:])

    return pd.concat(train_frames), pd.concat(
        test_frames
    ) if test_frames else pd.DataFrame()


def _select_forecast_value_column(forecast_df: pd.DataFrame, target_col: str) -> str:
    """Infer the forecast value column from common Nixtla outputs."""
    candidate_order = [
        "TimeGPT",
        target_col,
        f"{target_col}_hat",
        "forecast",
        "yhat",
    ]
    for col in candidate_order:
        if col in forecast_df.columns:
            return col
    raise ValueError(
        f"Could not find forecast value column in {forecast_df.columns.tolist()}"
    )


def finetune_timegpt(
    model_path: str,
    dataset_name: str,
    target_col: str,
    time_col: str,
    id_col: str,
    resolution_min: int,
    forecast_horizon: int,
    finetune_steps: int,
    freq: Optional[str] = None,
    api_key: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Dict:
    """
    Finetune TimeGPT via NixtlaClient.forecast and return simple metrics.
    """
    client = NixtlaClient(api_key=api_key)
    info_print("Loading processed series from cache...")
    series_df = _load_processed_series(
        dataset_name=dataset_name,
        target_col=target_col,
        time_col=time_col,
        id_col=id_col,
        resolution_min=resolution_min,
        freq=freq,
        min_length=forecast_horizon + 1,
    )

    info_print("Splitting into train/test per series...")
    train_df, test_df = _split_train_test(
        series_df, id_col=id_col, time_col=time_col, horizon=forecast_horizon
    )

    forecast_kwargs = dict(
        df=train_df[[id_col, time_col, target_col]],
        h=forecast_horizon,
        finetune_steps=finetune_steps,
        time_col=time_col,
        target_col=target_col,
    )
    if id_col:
        forecast_kwargs["id_col"] = id_col
    if freq:
        forecast_kwargs["freq"] = freq
    if model_path:
        forecast_kwargs["model"] = model_path

    info_print(
        f"Calling NixtlaClient.forecast with horizon={forecast_horizon}, "
        f"finetune_steps={finetune_steps}, freq={freq or 'auto'}"
    )
    forecast_df = client.forecast(**forecast_kwargs)
    value_col = _select_forecast_value_column(forecast_df, target_col)

    metrics = {"mae": None, "mse": None, "rmse": None, "count": 0}
    if not test_df.empty:
        merged = test_df.merge(
            forecast_df[[id_col, time_col, value_col]],
            on=[id_col, time_col],
            how="inner",
        )
        if merged.empty:
            info_print(
                "No overlap between forecast timestamps and test horizon; skipping metrics"
            )
        else:
            diff = merged[target_col] - merged[value_col]
            metrics["mae"] = float(np.abs(diff).mean())
            metrics["mse"] = float(np.square(diff).mean())
            metrics["rmse"] = float(np.sqrt(np.square(diff).mean()))
            metrics["count"] = int(len(merged))
            info_print(
                f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} "
                f"on {metrics['count']} rows"
            )
    else:
        info_print("Test split is empty; metrics will be None")

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = Path(run_dir) / "timegpt_forecast.csv"
        metrics_path = Path(run_dir) / "timegpt_metrics.json"
        forecast_df.to_csv(forecast_path, index=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        info_print(f"Saved forecast to {forecast_path}")
        info_print(f"Saved metrics to {metrics_path}")

    return metrics


def main():
    args = parse_arguments()

    config = {}
    if args.config:
        info_print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        if args.run_dir:
            run_dir = Path(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_copy_path = run_dir / "experiment_config.yaml"
            with open(config_copy_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            info_print(f"Configuration saved to: {config_copy_path}")

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    run_dir_path = Path(args.run_dir) if args.run_dir else None
    metrics = finetune_timegpt(
        model_path=model_config.get("path", "timegpt-1"),
        dataset_name=data_config.get("source_name", "kaggle_brisT1D"),
        target_col=data_config.get("target_col", "bg_mM"),
        time_col=data_config.get("time_col", "datetime"),
        id_col=data_config.get("id_col", "id"),
        resolution_min=data_config.get("resolution_min", 5),
        forecast_horizon=training_config.get("forecast_horizon", 12),
        finetune_steps=training_config.get("finetune_steps", 10),
        freq=training_config.get("freq"),
        api_key=model_config.get("api_key", os.environ.get("NIXTLA_API_KEY")),
        run_dir=run_dir_path,
    )

    if args.run_dir:
        metrics_file = Path(args.run_dir) / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        info_print(f"Training metrics saved to: {metrics_file}")
        print(f"METRICS_JSON:{json.dumps(metrics)}")


if __name__ == "__main__":
    main()
