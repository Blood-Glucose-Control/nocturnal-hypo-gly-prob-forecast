"""Evaluation and plotting helpers for forecasting workflow phases."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data.versioning.dataset_registry import DatasetRegistry
from ...evaluation.episode_builders import build_midnight_episodes

logger = logging.getLogger(__name__)


class ForecastValidationError(RuntimeError):
    """Raised when generated forecasts fail sanity validation."""


def _validate_forecast_values(
    raw_predictions: Any,
    *,
    phase_name: str,
    dataset_name: str,
    patient_id: Any,
) -> np.ndarray:
    """Validate and normalize model forecast outputs for downstream workflow steps."""
    predictions = np.atleast_1d(np.asarray(raw_predictions, dtype=float).squeeze())
    if predictions.size == 0:
        raise ForecastValidationError(
            f"Empty forecast array for phase={phase_name}, dataset={dataset_name}, "
            f"patient={patient_id}."
        )

    finite_mask = np.isfinite(predictions)
    finite_count = int(finite_mask.sum())
    non_finite_count = int(predictions.size - finite_count)
    if finite_count == 0:
        nan_count = int(np.isnan(predictions).sum())
        inf_count = int(np.isinf(predictions).sum())
        raise ForecastValidationError(
            f"All predictions are non-finite for phase={phase_name}, "
            f"dataset={dataset_name}, patient={patient_id} "
            f"(nan={nan_count}, inf={inf_count}, total={predictions.size})."
        )

    if non_finite_count > 0:
        logger.warning(
            "Detected %d/%d non-finite predictions for phase=%s, dataset=%s, "
            "patient=%s; proceeding with remaining finite values.",
            non_finite_count,
            predictions.size,
            phase_name,
            dataset_name,
            patient_id,
        )

    return predictions


def _generate_forecasts(
    model,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    model_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate per-dataset forecasting outputs for a workflow phase."""
    logger.info(f"  Generating forecasts for phase: {phase_name}")
    try:
        context_length = model.config.context_length
        forecast_length = model.config.forecast_length
        registry = DatasetRegistry(holdout_config_dir=config_dir)

        if model_config_overrides:
            input_features = model_config_overrides.get("input_features") or []
            target_features = model_config_overrides.get("target_features") or []
            if input_features or target_features:
                model_features = list(input_features) + list(target_features)
                logger.info(f"  Using model config features: {model_features}")
            else:
                model_features = None
        else:
            model_features = None

        predictions_dir = Path(output_dir) / "predictions" / phase_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Predictions output directory: {predictions_dir}")

        forecast_results = {}
        for dataset_name in dataset_names:
            logger.info(f"  --- Generating forecast for dataset: {dataset_name} ---")
            holdout_data = registry.load_holdout_data_only(dataset_name)
            patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
            valid_patients = holdout_data[patient_col].dropna()
            if valid_patients.empty:
                logger.warning(
                    f"  No valid patient IDs found in holdout data for {dataset_name}. "
                    f"All {len(holdout_data)} rows have NaN {patient_col}. Skipping."
                )
                continue

            first_patient = valid_patients.iloc[0]
            patient_data = holdout_data[holdout_data[patient_col] == first_patient]
            logger.info(f"  First holdout patient: {first_patient}")
            logger.info(f"  Patient data shape: {patient_data.shape}")

            if model_features:
                required_cols = ["p_num", "id", "datetime"]
                forecast_cols = [
                    col
                    for col in model_features + required_cols
                    if col in patient_data.columns
                ]
            else:
                forecast_cols = [
                    col for col in training_columns if col in patient_data.columns
                ]

            glucose_col = "bg_mM"
            patient_ts = patient_data.copy()
            patient_ts["datetime"] = pd.to_datetime(patient_ts["datetime"])
            patient_ts = patient_ts.set_index("datetime").sort_index()

            episodes, ep_stats = build_midnight_episodes(
                patient_ts,
                context_length=context_length,
                forecast_length=forecast_length,
                target_col=glucose_col,
            )
            if not episodes:
                logger.warning(
                    f"  No valid midnight episodes for patient {first_patient} "
                    f"in {dataset_name} (checked {ep_stats['total_anchors']} "
                    f"anchors, {ep_stats['skipped_bg_nan']} had BG gaps). Skipping."
                )
                continue

            episode = episodes[0]
            logger.info(
                f"  Using midnight episode anchored at {episode['anchor']} "
                f"({len(episodes)} valid, {ep_stats['skipped_bg_nan']} skipped)"
            )

            historical_glucose = episode["context_df"][glucose_col].values
            actual_glucose = episode["target_bg"]

            context_data = episode["context_df"].reset_index()
            context_data.rename(
                columns={context_data.columns[0]: "datetime"}, inplace=True
            )
            context_data[patient_col] = first_patient
            for col in forecast_cols:
                if col not in context_data.columns and col in patient_ts.columns:
                    context_data[col] = context_data["datetime"].map(patient_ts[col])

            context_data = context_data[
                [col for col in context_data.columns if col not in ["source_dataset"]]
            ]
            logger.info(f"  Context data shape: {context_data.shape}")

            predictions_raw = model.predict(context_data)
            logger.info(f"    Raw predictions shape: {predictions_raw.shape}")

            if len(predictions_raw.shape) == 3:
                extracted_predictions = predictions_raw[0, :, 0]
            elif len(predictions_raw.shape) == 2:
                extracted_predictions = predictions_raw[:, 0]
            else:
                extracted_predictions = predictions_raw.squeeze()

            predictions = _validate_forecast_values(
                extracted_predictions,
                phase_name=phase_name,
                dataset_name=dataset_name,
                patient_id=first_patient,
            )

            logger.info(f"    Extracted glucose predictions shape: {predictions.shape}")
            forecast_datetimes = pd.date_range(
                episode["anchor"], periods=forecast_length, freq="5min"
            ).values

            forecast_results[dataset_name] = {
                "predictions": predictions,
                "historical_glucose": historical_glucose,
                "actual_glucose": actual_glucose,
                "patient_id": first_patient,
                "context_length": context_length,
                "forecast_length": forecast_length,
                "forecast_datetimes": forecast_datetimes,
            }
            logger.info(f"  ✓ Generated forecast for {dataset_name}")
            logger.info(f"    Glucose predictions preview (first 5): {predictions[:5]}")

            predictions_json = (
                predictions_dir
                / f"{phase_name}_{dataset_name}_patient{first_patient}.json"
            )
            predictions_data = {
                "phase": phase_name,
                "dataset": dataset_name,
                "patient_id": str(first_patient),
                "raw_predictions_shape": list(predictions_raw.shape),
                "glucose_predictions_shape": list(predictions.shape),
                "glucose_predictions": predictions.tolist(),
                "finite_prediction_count": int(np.isfinite(predictions).sum()),
                "non_finite_prediction_count": int(
                    predictions.size - np.isfinite(predictions).sum()
                ),
                "forecast_length": forecast_length,
                "context_length": context_length,
            }
            if forecast_datetimes is not None:
                predictions_data["forecast_datetimes"] = [
                    str(dt) for dt in forecast_datetimes
                ]
            with open(predictions_json, "w") as f:
                json.dump(predictions_data, f, indent=2)
            logger.info(f"    ✓ Predictions saved to: {predictions_json}")

        logger.info(f"  ✓ Forecast generation completed for phase: {phase_name}")
        return forecast_results
    except Exception as e:
        logger.error(f"  ✗ Failed to generate forecasts: {e}")
        traceback.print_exc()
        raise


def _plot_forecasts(
    forecast_results: Optional[dict],
    output_dir: str,
    phase_name: str,
) -> bool:
    """Create phase forecast plots and CSV summaries."""
    logger.info(f"  Plotting forecasts for phase: {phase_name}")
    if forecast_results is None:
        logger.error("  ✗ No forecast results to plot")
        return False
    try:
        forecast_dir = Path(output_dir) / "forecasts" / phase_name
        forecast_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Forecast output directory: {forecast_dir}")

        for dataset_name, results in forecast_results.items():
            logger.info(f"  --- Plotting forecast for dataset: {dataset_name} ---")
            predictions = _validate_forecast_values(
                results["predictions"],
                phase_name=phase_name,
                dataset_name=dataset_name,
                patient_id=results["patient_id"],
            )
            finite_predictions = predictions[np.isfinite(predictions)]
            historical_glucose = results["historical_glucose"]
            actual_glucose = results["actual_glucose"]
            patient_id = results["patient_id"]
            context_length = results["context_length"]
            forecast_datetimes = results.get("forecast_datetimes")

            logger.info(f"    Predictions shape for plotting: {predictions.shape}")
            logger.info(
                f"    Predictions range: "
                f"[{finite_predictions.min():.2f}, {finite_predictions.max():.2f}]"
            )

            _, ax = plt.subplots(figsize=(15, 6))
            use_datetime_axis = (
                forecast_datetimes is not None and len(forecast_datetimes) > 0
            )
            forecast_dts = pd.DatetimeIndex([])
            historical_dts = pd.DatetimeIndex([])
            if use_datetime_axis:
                forecast_dts = pd.to_datetime(cast(Any, forecast_datetimes))
                time_delta = pd.Timedelta(minutes=5)
                historical_dts = pd.date_range(
                    end=forecast_dts[0] - time_delta,
                    periods=len(historical_glucose),
                    freq="5min",
                )
                ax.plot(
                    historical_dts,
                    historical_glucose,
                    "b-",
                    label="Historical Data",
                    linewidth=2,
                )
                actual_dts = forecast_dts[: len(actual_glucose)]
                ax.plot(actual_dts, actual_glucose, "g-", label="Actual", linewidth=2)
                forecast_dts_pred = forecast_dts[: len(predictions)]
                ax.plot(
                    forecast_dts_pred, predictions, "r--", label="Forecast", linewidth=2
                )
                ax.axvline(
                    x=forecast_dts[0],
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    label="Forecast Start",
                )
                import matplotlib.dates as mdates

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=45, ha="right")
                ax.set_xlabel("Time", fontsize=12)
            else:
                historical_time = np.arange(len(historical_glucose))
                ax.plot(
                    historical_time,
                    historical_glucose,
                    "b-",
                    label="Historical Data",
                    linewidth=2,
                )
                actual_time = np.arange(
                    len(historical_glucose),
                    len(historical_glucose) + len(actual_glucose),
                )
                ax.plot(actual_time, actual_glucose, "g-", label="Actual", linewidth=2)
                forecast_time = np.arange(
                    len(historical_glucose),
                    len(historical_glucose) + len(predictions),
                )
                ax.plot(
                    forecast_time, predictions, "r--", label="Forecast", linewidth=2
                )
                ax.axvline(
                    x=len(historical_glucose),
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    label="Forecast Start",
                )
                ax.set_xlabel("Time Steps", fontsize=12)

            ax.axhline(
                y=3.9,
                color="orange",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Hypoglycemia (3.9 mM)",
            )
            ax.axhline(
                y=10.0,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Hyperglycemia (10.0 mM)",
            )
            ax.set_ylabel("Blood Glucose (mM)", fontsize=12)
            ax.set_title(
                f"[{phase_name.upper()}] Blood Glucose Forecast - {dataset_name} "
                f"(Context: {context_length}, Patient: {patient_id})",
                fontsize=14,
            )
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            plot_path = forecast_dir / f"{phase_name}_{dataset_name}_forecast.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"    ✓ Forecast plot saved to: {plot_path}")
            plt.close()

            forecast_csv_path = forecast_dir / f"{phase_name}_{dataset_name}_data.csv"
            if use_datetime_axis:
                all_datetimes = list(historical_dts) + list(
                    forecast_dts[: len(actual_glucose)]
                )
                forecast_data_df = pd.DataFrame(
                    {
                        "datetime": all_datetimes,
                        "historical": list(historical_glucose)
                        + [np.nan] * len(actual_glucose),
                        "actual": [np.nan] * len(historical_glucose)
                        + list(actual_glucose),
                        "forecast": [np.nan] * len(historical_glucose)
                        + list(predictions),
                    }
                )
            else:
                historical_time = np.arange(len(historical_glucose))
                actual_time = np.arange(
                    len(historical_glucose),
                    len(historical_glucose) + len(actual_glucose),
                )
                forecast_data_df = pd.DataFrame(
                    {
                        "time_step": list(historical_time) + list(actual_time),
                        "historical": list(historical_glucose)
                        + [np.nan] * len(actual_glucose),
                        "actual": [np.nan] * len(historical_glucose)
                        + list(actual_glucose),
                        "forecast": [np.nan] * len(historical_glucose)
                        + list(predictions),
                    }
                )
            forecast_data_df.to_csv(forecast_csv_path, index=False)
            logger.info(f"    ✓ Forecast data saved to: {forecast_csv_path}")

        logger.info(f"  ✓ Forecast plotting completed for phase: {phase_name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to plot forecasts: {e}")
        traceback.print_exc()
        raise


def evaluate_and_plot(
    model,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    model_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate forecasts and plots for a workflow phase."""
    logger.info("-" * 40)
    logger.info(f"Evaluating and plotting for phase: {phase_name}")
    logger.info("-" * 40)
    forecast_results = _generate_forecasts(
        model=model,
        training_columns=training_columns,
        dataset_names=dataset_names,
        config_dir=config_dir,
        output_dir=output_dir,
        phase_name=phase_name,
        model_config_overrides=model_config_overrides,
    )
    if not forecast_results:
        raise ForecastValidationError(
            f"No forecasts were generated for phase={phase_name}. "
            f"Datasets requested: {dataset_names}."
        )

    _plot_forecasts(
        forecast_results=forecast_results,
        output_dir=output_dir,
        phase_name=phase_name,
    )
    return forecast_results
