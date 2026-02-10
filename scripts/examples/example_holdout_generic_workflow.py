#!/usr/bin/env python3
"""
End-to-end example: Holdout system with GENERIC model training and evaluation.

This script demonstrates a complete workflow that can work with different
time series foundation models (TTM, Chronos, Moment, etc.) by using the
base class interfaces.

This is a refactored version of example_holdout_ttm_workflow.py that:
- Supports multiple model types via --model-type argument
- Uses generic base class interfaces (BaseTimeSeriesFoundationModel, ModelConfig)
- Can be extended to support new model types by implementing the model factory

Workflow Steps:
  1. Check holdout configs exist
  2. Validate holdout configs
  3. Load and combine training data from multiple datasets
  4. Zero-shot evaluation (pretrained model, no fine-tuning)
  5. Fine-tune model for specified epochs
  6. Load model from checkpoint (verify save/load works)
  7. Resume training on loaded model
  8. Full holdout evaluation on all datasets

Usage:
    # Run with default settings (TTM model)
    python scripts/examples/example_holdout_generic_workflow.py --datasets lynch_2022 brown_2019

    # Run with specific model type
    python scripts/examples/example_holdout_generic_workflow.py --model-type ttm --datasets lynch_2022 aleppo_2017

    # Skip training (only zero-shot evaluation)
    python scripts/examples/example_holdout_generic_workflow.py --model-type ttm --datasets lynch_2022 --skip-training

    # Use shell wrapper for local execution:
    ./scripts/examples/run_holdout_generic_workflow.sh
"""

import argparse
import json
import logging
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.dataset_combiner import (
    combine_datasets_for_training,
    print_dataset_column_table,
)
from src.data.preprocessing.imputation import impute_missing_values
from src.models.base import DistributedConfig, GPUManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from data processing modules
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)
logging.getLogger("src.models").setLevel(logging.WARNING)
logging.getLogger("src.utils").setLevel(logging.WARNING)


# =============================================================================
# MODEL FACTORY: Generic model creation based on model type
# =============================================================================

# Registry of supported model types
SUPPORTED_MODELS = {}


def register_model(model_type: str):
    """Decorator to register a model type in the factory."""

    def decorator(cls):
        SUPPORTED_MODELS[model_type] = cls
        return cls

    return decorator


@dataclass
class GenericModelConfig:
    """Generic configuration that works across all model types.

    This wraps the model-specific config and provides a unified interface.
    """

    model_type: str
    model_path: str
    context_length: int = 512
    forecast_length: int = 96
    batch_size: int = 2048
    num_epochs: int = 1
    training_mode: str = "fine_tune"  # "zero_shot", "fine_tune", "from_scratch"
    freeze_backbone: bool = False
    use_cpu: bool = False
    fp16: bool = True
    learning_rate: float = 1e-4

    # Additional model-specific config can be passed as kwargs
    extra_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_config is None:
            self.extra_config = {}


def load_model_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load model configuration from a YAML file.

    Reads a YAML model config file (e.g., configs/models/ttm/default.yaml)
    and returns it as a dictionary. These values are used to configure the
    model-specific config (TTMConfig, ChronosConfig, etc.).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration parameters from the YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        logger.warning(f"Model config file is empty: {config_path}")
        return {}

    logger.info(f"Loaded model config from: {config_path}")
    logger.info(f"  Parameters specified: {len(config)}")
    for key, value in config.items():
        logger.info(f"    {key}: {value}")

    return config


class ModelFactory:
    """Factory for creating model instances based on model type."""

    @staticmethod
    def get_default_model_path(model_type: str) -> str:
        """Get the default model path for a given model type."""
        defaults = {
            "ttm": "ibm-granite/granite-timeseries-ttm-r2",
            "chronos": "amazon/chronos-t5-small",
            "moment": "AutonLab/MOMENT-1-small",
            # Add more defaults as models are implemented
        }
        return defaults.get(model_type, "")

    @staticmethod
    def create_model(
        config: GenericModelConfig,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """Create a model instance based on the configuration.

        Args:
            config: Generic model configuration
            distributed_config: Optional distributed training configuration

        Returns:
            Model instance (type depends on model_type)

        Raises:
            ValueError: If model_type is not supported
        """
        model_type = config.model_type.lower()

        if model_type == "ttm":
            return ModelFactory._create_ttm_model(config, distributed_config)
        elif model_type == "chronos":
            return ModelFactory._create_chronos_model(config, distributed_config)
        elif model_type == "moment":
            return ModelFactory._create_moment_model(config, distributed_config)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: ttm, chronos, moment"
            )

    @staticmethod
    def _create_ttm_model(
        config: GenericModelConfig,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """Create a TTM model instance."""
        from src.models.ttm import TTMForecaster, TTMConfig

        ttm_config = TTMConfig(
            model_path=config.model_path,
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            training_mode=config.training_mode,
            freeze_backbone=config.freeze_backbone,
            use_cpu=config.use_cpu,
            fp16=config.fp16,
            learning_rate=config.learning_rate,
            **config.extra_config,
        )

        return TTMForecaster(ttm_config, distributed_config=distributed_config)

    @staticmethod
    def _create_chronos_model(
        config: GenericModelConfig,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """Create a Chronos model instance."""
        # Import when needed to avoid dependency issues
        try:
            from src.models.chronos import ChronosForecaster, ChronosConfig

            chronos_config = ChronosConfig(
                model_path=config.model_path,
                context_length=config.context_length,
                forecast_length=config.forecast_length,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                training_mode=config.training_mode,
                use_cpu=config.use_cpu,
                fp16=config.fp16,
                **config.extra_config,
            )

            return ChronosForecaster(
                chronos_config, distributed_config=distributed_config
            )
        except ImportError as e:
            raise ImportError(
                f"Chronos model not available. Install chronos dependencies: {e}"
            )

    @staticmethod
    def _create_moment_model(
        config: GenericModelConfig,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """Create a MOMENT model instance."""
        try:
            from src.models.moment import MomentForecaster, MomentConfig

            moment_config = MomentConfig(
                model_path=config.model_path,
                context_length=config.context_length,
                forecast_length=config.forecast_length,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                training_mode=config.training_mode,
                use_cpu=config.use_cpu,
                fp16=config.fp16,
                **config.extra_config,
            )

            return MomentForecaster(
                moment_config, distributed_config=distributed_config
            )
        except ImportError as e:
            raise ImportError(
                f"MOMENT model not available. Install moment dependencies: {e}"
            )

    @staticmethod
    def create_zero_shot_config(
        model_type: str,
        model_path: Optional[str] = None,
        context_length: int = 512,
        forecast_length: int = 96,
        batch_size: int = 2048,
        use_cpu: bool = False,
        fp16: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> GenericModelConfig:
        """Create a configuration for zero-shot evaluation.

        Args:
            model_type: Type of model (ttm, chronos, moment)
            model_path: Path to pretrained model (uses default if None)
            context_length: Number of historical time steps
            forecast_length: Number of future time steps to predict
            batch_size: Batch size for inference
            use_cpu: Force CPU usage
            fp16: Use mixed precision
            extra_config: Additional model-specific config from YAML file

        Returns:
            GenericModelConfig configured for zero-shot evaluation
        """
        # Start with YAML config as base, then override with explicit args
        yaml_config = dict(extra_config) if extra_config else {}

        # Extract top-level params from YAML (CLI args override these)
        resolved_model_path = model_path or yaml_config.pop(
            "model_path", ModelFactory.get_default_model_path(model_type)
        )
        resolved_context = yaml_config.pop("context_length", context_length)
        resolved_forecast = yaml_config.pop("forecast_length", forecast_length)
        resolved_batch = yaml_config.pop("batch_size", batch_size)

        # Remove params that are forced for zero-shot
        for key in [
            "num_epochs",
            "training_mode",
            "freeze_backbone",
            "use_cpu",
            "fp16",
            "learning_rate",
        ]:
            yaml_config.pop(key, None)

        return GenericModelConfig(
            model_type=model_type,
            model_path=resolved_model_path,
            context_length=resolved_context,
            forecast_length=resolved_forecast,
            batch_size=resolved_batch,
            num_epochs=0,
            training_mode="zero_shot",
            freeze_backbone=True,
            use_cpu=use_cpu,
            fp16=fp16,
            extra_config=yaml_config,  # Remaining TTM-specific params
        )

    @staticmethod
    def create_finetune_config(
        model_type: str,
        model_path: Optional[str] = None,
        context_length: int = None,
        forecast_length: int = None,
        batch_size: int = None,
        num_epochs: int = None,
        learning_rate: float = None,
        use_cpu: bool = False,
        fp16: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> GenericModelConfig:
        """Create a configuration for fine-tuning.

        Parameters are resolved in priority order: CLI args > YAML config > defaults.

        Args:
            model_type: Type of model (ttm, chronos, moment)
            model_path: Path to pretrained model (uses YAML or default if None)
            context_length: Number of historical time steps (CLI override)
            forecast_length: Number of future time steps to predict (CLI override)
            batch_size: Batch size for training (CLI override)
            num_epochs: Number of training epochs (CLI override)
            learning_rate: Learning rate (CLI override)
            use_cpu: Force CPU usage
            fp16: Use mixed precision
            extra_config: Additional model-specific config from YAML file

        Returns:
            GenericModelConfig configured for fine-tuning
        """
        # Start with YAML config as base
        yaml_config = dict(extra_config) if extra_config else {}

        # Resolve top-level params: CLI arg > YAML > default
        resolved_model_path = model_path or yaml_config.pop(
            "model_path", ModelFactory.get_default_model_path(model_type)
        )
        resolved_context = (
            context_length
            if context_length is not None
            else yaml_config.pop("context_length", 512)
        )
        resolved_forecast = (
            forecast_length
            if forecast_length is not None
            else yaml_config.pop("forecast_length", 96)
        )
        resolved_batch = (
            batch_size
            if batch_size is not None
            else yaml_config.pop("batch_size", 2048)
        )
        resolved_epochs = (
            num_epochs if num_epochs is not None else yaml_config.pop("num_epochs", 1)
        )
        resolved_lr = (
            learning_rate
            if learning_rate is not None
            else yaml_config.pop("learning_rate", 1e-4)
        )
        resolved_mode = yaml_config.pop("training_mode", "fine_tune")
        resolved_freeze = yaml_config.pop("freeze_backbone", False)

        # Remove hardware params from extra_config (handled by CLI)
        yaml_config.pop("use_cpu", None)
        yaml_config.pop("fp16", None)

        return GenericModelConfig(
            model_type=model_type,
            model_path=resolved_model_path,
            context_length=resolved_context,
            forecast_length=resolved_forecast,
            batch_size=resolved_batch,
            num_epochs=resolved_epochs,
            training_mode=resolved_mode,
            freeze_backbone=resolved_freeze,
            use_cpu=use_cpu,
            fp16=fp16,
            learning_rate=resolved_lr,
            extra_config=yaml_config,  # Remaining model-specific params
        )

    @staticmethod
    def load_model(
        model_type: str,
        model_path: str,
        config: GenericModelConfig,
    ):
        """Load a model from a checkpoint.

        Args:
            model_type: Type of model (ttm, chronos, moment)
            model_path: Path to the saved model checkpoint
            config: GenericModelConfig used for loading

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model_type is not supported
        """
        model_type_lower = model_type.lower()

        if model_type_lower == "ttm":
            from src.models.ttm import TTMForecaster, TTMConfig

            ttm_config = TTMConfig(
                model_path=config.model_path,
                context_length=config.context_length,
                forecast_length=config.forecast_length,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                training_mode=config.training_mode,
                freeze_backbone=config.freeze_backbone,
                use_cpu=config.use_cpu,
                fp16=config.fp16,
                learning_rate=config.learning_rate,
                **config.extra_config,
            )
            return TTMForecaster.load(model_path, ttm_config)
        elif model_type_lower == "chronos":
            from src.models.chronos import ChronosForecaster, ChronosConfig

            chronos_config = ChronosConfig(
                model_path=config.model_path,
                context_length=config.context_length,
                forecast_length=config.forecast_length,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                training_mode=config.training_mode,
                use_cpu=config.use_cpu,
                fp16=config.fp16,
                **config.extra_config,
            )
            return ChronosForecaster.load(model_path, chronos_config)
        elif model_type_lower == "moment":
            from src.models.moment import MomentForecaster, MomentConfig

            moment_config = MomentConfig(
                model_path=config.model_path,
                context_length=config.context_length,
                forecast_length=config.forecast_length,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                training_mode=config.training_mode,
                use_cpu=config.use_cpu,
                fp16=config.fp16,
                **config.extra_config,
            )
            return MomentForecaster.load(model_path, moment_config)
        else:
            raise ValueError(
                f"Unsupported model type for loading: {model_type}. "
                f"Supported types: ttm, chronos, moment"
            )


# =============================================================================
# HELPER FUNCTIONS: Prediction Generation and Plotting
# =============================================================================


def _generate_forecasts(
    model,  # BaseTimeSeriesFoundationModel or compatible
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    zero_shot: bool = False,
) -> Optional[Dict[str, Dict]]:
    """Helper: Generate forecasts using the model and holdout data.

    Args:
        model: Forecaster model (TTM, Chronos, Moment, etc.)
        training_columns: List of column names used during training
        dataset_names: List of dataset names to generate forecasts for
        config_dir: Directory containing holdout configurations
        output_dir: Directory where prediction files will be saved
        phase_name: Identifier for this phase (e.g., "zero_shot", "after_training")
        zero_shot: If True, use predict_zero_shot() instead of predict()

    Returns:
        Dict[str, Dict]: Dictionary mapping dataset names to forecast results
    """
    logger.info(f"  Generating forecasts for phase: {phase_name}")

    try:
        context_length = model.config.context_length
        forecast_length = model.config.forecast_length
        registry = DatasetRegistry(holdout_config_dir=config_dir)

        # Create predictions output directory with phase identifier
        predictions_dir = Path(output_dir) / "predictions" / phase_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Predictions output directory: {predictions_dir}")

        forecast_results = {}

        # Process each dataset's first holdout patient
        for dataset_name in dataset_names:
            logger.info(f"  --- Generating forecast for dataset: {dataset_name} ---")

            # Load holdout data
            holdout_data = registry.load_holdout_data_only(dataset_name)

            # Get first patient (skip NaN p_num rows from gap-fill)
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

            forecast_cols = [
                col for col in training_columns if col in patient_data.columns
            ]
            # Slice to get context + forecast length
            total_length = context_length + forecast_length
            forecast_data = patient_data.iloc[:total_length][forecast_cols].copy()

            logger.info(f"  Forecast data shape: {forecast_data.shape}")

            # Keep necessary columns for preprocessing (p_num, datetime)
            # Only remove source_dataset if present
            exclude_cols = ["source_dataset"]
            forecast_cols_for_model = [
                col for col in forecast_data.columns if col not in exclude_cols
            ]
            forecast_data_for_model = forecast_data[forecast_cols_for_model].copy()

            # Generate predictions
            if zero_shot:
                predictions_raw = model.predict_zero_shot(forecast_data_for_model)
            else:
                predictions_raw = model.predict(forecast_data_for_model)

            # TTM returns predictions in shape (samples, forecast_length, num_channels)
            # For univariate glucose prediction, we need channel 0
            logger.info(f"    Raw predictions shape: {predictions_raw.shape}")

            if len(predictions_raw.shape) == 3:
                predictions = predictions_raw[0, :, 0]
            elif len(predictions_raw.shape) == 2:
                predictions = predictions_raw[:, 0]
            else:
                predictions = predictions_raw.squeeze()

            logger.info(f"    Extracted glucose predictions shape: {predictions.shape}")

            # Extract glucose values
            glucose_col = "bg_mM"
            historical_glucose = forecast_data[glucose_col].values[:context_length]
            actual_glucose = forecast_data[glucose_col].values[context_length:]

            # Extract datetime values for the forecast period
            datetime_col = "datetime"
            if datetime_col in forecast_data.columns:
                forecast_datetimes = forecast_data[datetime_col].values[
                    context_length : context_length + forecast_length
                ]
            else:
                forecast_datetimes = None

            # Store results
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

            # Save predictions to JSON for quick inspection
            predictions_json = (
                predictions_dir
                / f"{phase_name}_{dataset_name}_patient{first_patient}.json"
            )

            # Prepare data for JSON serialization
            predictions_data = {
                "phase": phase_name,
                "dataset": dataset_name,
                "patient_id": str(first_patient),
                "raw_predictions_shape": list(predictions_raw.shape),
                "glucose_predictions_shape": list(predictions.shape),
                "glucose_predictions": predictions.tolist(),
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
        return None


def _plot_forecasts(
    forecast_results: dict,
    output_dir: str,
    phase_name: str,
) -> bool:
    """Helper: Create plots and save forecast visualizations.

    Args:
        forecast_results: Dictionary from _generate_forecasts containing forecast data
        output_dir: Directory where plots will be saved
        phase_name: Identifier for this phase (e.g., "zero_shot", "after_training")

    Returns:
        bool: True if plotting succeeded, False otherwise
    """
    logger.info(f"  Plotting forecasts for phase: {phase_name}")

    if forecast_results is None:
        logger.error("  ✗ No forecast results to plot")
        return False

    try:
        # Create forecast output directory with phase identifier
        forecast_dir = Path(output_dir) / "forecasts" / phase_name
        forecast_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Forecast output directory: {forecast_dir}")

        # Plot each dataset's forecast
        for dataset_name, results in forecast_results.items():
            logger.info(f"  --- Plotting forecast for dataset: {dataset_name} ---")

            predictions = results["predictions"]
            historical_glucose = results["historical_glucose"]
            actual_glucose = results["actual_glucose"]
            patient_id = results["patient_id"]
            context_length = results["context_length"]
            forecast_datetimes = results.get("forecast_datetimes")

            # Ensure predictions is 1D for plotting
            predictions = np.array(predictions).squeeze()
            logger.info(f"    Predictions shape for plotting: {predictions.shape}")
            logger.info(
                f"    Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]"
            )

            # Create plot
            fig, ax = plt.subplots(figsize=(15, 6))

            # Determine if we can use datetime x-axis
            use_datetime_axis = (
                forecast_datetimes is not None and len(forecast_datetimes) > 0
            )

            if use_datetime_axis:
                # Convert to pandas datetime for plotting
                forecast_dts = pd.to_datetime(forecast_datetimes)

                # Estimate historical datetimes by subtracting from first forecast time
                # Assuming 5-minute intervals (common for CGM data)
                time_delta = pd.Timedelta(minutes=5)
                historical_dts = pd.date_range(
                    end=forecast_dts[0] - time_delta,
                    periods=len(historical_glucose),
                    freq="5min",
                )

                # Plot historical data
                ax.plot(
                    historical_dts,
                    historical_glucose,
                    "b-",
                    label="Historical Data",
                    linewidth=2,
                )

                # Plot actual future values
                actual_dts = forecast_dts[: len(actual_glucose)]
                ax.plot(actual_dts, actual_glucose, "g-", label="Actual", linewidth=2)

                # Plot forecast
                forecast_dts_pred = forecast_dts[: len(predictions)]
                ax.plot(
                    forecast_dts_pred, predictions, "r--", label="Forecast", linewidth=2
                )

                # Add vertical line at forecast start
                ax.axvline(
                    x=forecast_dts[0],
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    label="Forecast Start",
                )

                # Format x-axis for datetime
                import matplotlib.dates as mdates

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=45, ha="right")
                ax.set_xlabel("Time", fontsize=12)
            else:
                # Fallback to integer time steps
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
                    len(historical_glucose), len(historical_glucose) + len(predictions)
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

            # Add reference lines for hypo/hyper thresholds (in mM)
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

            # Labels and title
            ax.set_ylabel("Blood Glucose (mM)", fontsize=12)
            ax.set_title(
                f"[{phase_name.upper()}] Blood Glucose Forecast - {dataset_name} "
                f"(Context: {context_length}, Patient: {patient_id})",
                fontsize=14,
            )
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Save plot
            plot_path = forecast_dir / f"{phase_name}_{dataset_name}_forecast.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"    ✓ Forecast plot saved to: {plot_path}")

            plt.close()

            # Save forecast data to CSV
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
        return False


def _evaluate_and_plot(
    model,  # BaseTimeSeriesFoundationModel or compatible
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    zero_shot: bool = False,
) -> Optional[Dict]:
    """Helper: Generate forecasts and plots for a given phase.

    This is the main helper that combines forecast generation and plotting.
    Called after each major workflow phase (zero-shot, training, loading, etc.)

    Args:
        model: Forecaster model
        training_columns: List of column names from training data
        dataset_names: List of dataset names
        config_dir: Holdout config directory
        output_dir: Output directory for artifacts
        phase_name: Identifier for this phase
        zero_shot: If True, use predict_zero_shot() for inference

    Returns:
        dict: Forecast results, or None if failed
    """
    logger.info("-" * 40)
    logger.info(f"Evaluating and plotting for phase: {phase_name}")
    logger.info("-" * 40)

    # Generate forecasts
    forecast_results = _generate_forecasts(
        model=model,
        training_columns=training_columns,
        dataset_names=dataset_names,
        config_dir=config_dir,
        output_dir=output_dir,
        phase_name=phase_name,
        zero_shot=zero_shot,
    )

    # Plot forecasts
    if forecast_results is not None:
        _plot_forecasts(
            forecast_results=forecast_results,
            output_dir=output_dir,
            phase_name=phase_name,
        )

    return forecast_results


# =============================================================================
# STEP FUNCTIONS
# =============================================================================


def step1_generate_holdout_configs(
    config_dir: str = "configs/data/holdout",
    output_dir: str | None = None,
    datasets: list | None = None,
) -> bool:
    """Step 1: Generate holdout configurations and copy to artifacts directory."""
    logger.info("=" * 80)
    logger.info("STEP 1: Generate Holdout Configurations")
    logger.info("=" * 80)

    config_path = Path(config_dir)

    if config_path.exists():
        configs = list(config_path.glob("*.yaml"))
        logger.info(f"✓ Holdout configs already exist: {len(configs)} datasets")
        for cfg in configs:
            logger.info(f"  - {cfg.stem}")

        # Copy only configs for datasets being used in this run
        if output_dir and datasets:
            artifacts_config_dir = Path(output_dir) / "configs"
            artifacts_config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Copying configs to artifacts directory: \n\t {artifacts_config_dir}"
            )
            logger.info(f"Datasets in this run: {', '.join(datasets)}")

            copied_count = 0
            for cfg in configs:
                # Only copy if this config matches one of the datasets being used
                if cfg.stem in datasets:
                    dest = artifacts_config_dir / cfg.name
                    shutil.copy2(cfg, dest)
                    logger.info(f"  ✓ Copied: {cfg.name}")
                    copied_count += 1

            logger.info(
                f"✓ Copied {copied_count}/{len(datasets)} configs to: \n\t {artifacts_config_dir}"
            )

        return True
    else:
        logger.warning(f"⚠ Config directory does not exist: {config_dir}")
        logger.info(
            "  Run: python scripts/data_processing_scripts/generate_holdout_configs.py"
        )
        return False


def step2_validate_holdout_configs(datasets: list, config_dir: str) -> bool:
    """Step 2: Validate holdout configurations for all datasets with comprehensive checks."""
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 2: Validate Holdout Configurations")
    logger.info(f"Validating {len(datasets)} dataset(s)")
    logger.info("=" * 80)

    from src.data.versioning import holdout_utils

    registry = DatasetRegistry(holdout_config_dir=config_dir)

    # Validate each dataset and collect results
    validation_results = []
    for idx, dataset_name in enumerate(datasets):
        logger.info(" ")
        logger.info(f"--- Dataset {idx + 1}/{len(datasets)}: {dataset_name} ---")
        # Get config info
        config = registry.get_holdout_config(dataset_name)
        if config is None:
            logger.error(f"✗ No config found for {dataset_name}")
            validation_results.append(
                {
                    "dataset_name": dataset_name,
                    "config_exists": False,
                    "load_successful": False,
                    "no_data_leakage": False,
                    "train_size": 0,
                    "holdout_size": 0,
                    "errors": ["No holdout configuration found"],
                }
            )
            continue

        # Log config details
        logger.info(f"✓ Config loaded: {config.holdout_type.value}")
        if config.temporal_config:
            logger.info(
                f"  Temporal holdout: {config.temporal_config.holdout_percentage*100}%"
            )
        if config.patient_config:
            logger.info(
                f"  Holdout patients: {len(config.patient_config.holdout_patients)}"
            )

        # Run comprehensive validation (suppress verbose output, we're logging manually)
        results = holdout_utils.validate_holdout_config(
            dataset_name, registry, verbose=False
        )
        validation_results.append(results)

        # Log brief status
        if results["errors"]:
            logger.error(f"✗ Validation failed with {len(results['errors'])} error(s)")
            for error in results["errors"]:
                logger.error(f"    - {error}")
        else:
            logger.info("✓ All comprehensive validations passed")

    # Print summary table
    holdout_utils.print_validation_summary(validation_results, verbose=False)

    # Check if any failed
    failed_datasets = [r["dataset_name"] for r in validation_results if r["errors"]]
    if failed_datasets:
        logger.error(f"\n✗ Validation failed for: {', '.join(failed_datasets)}")
        return False

    logger.info("✓ All datasets validated successfully")
    return True


def step3_load_training_data(
    dataset_names: list, config_dir: str, output_dir: str = None
):
    """Step 3: Load and combine training data from multiple datasets."""
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 3: Load Training Data")
    logger.info("=" * 80)

    registry = DatasetRegistry(holdout_config_dir=config_dir)

    # Combine multiple datasets
    combined_data, column_info = combine_datasets_for_training(
        dataset_names=dataset_names, registry=registry, config_dir=config_dir
    )

    # Save split metadata (skipped/adjusted patients) to output dir
    split_metadata = registry.get_split_metadata()
    if split_metadata and output_dir:
        metadata_dir = Path(output_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "split_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(split_metadata, f, indent=2)
        logger.info(f"✓ Split metadata saved to: {metadata_path}")

        # Log summary
        for ds_name, meta in split_metadata.items():
            n_skipped = len(meta.get("skipped_patients", {}))
            n_adjusted = len(meta.get("adjusted_patients", {}))
            n_filled = meta.get("nan_p_num_filled", 0)
            if n_skipped or n_adjusted or n_filled:
                logger.info(
                    f"  {ds_name}: {n_skipped} skipped, {n_adjusted} adjusted, {n_filled:,} NaN p_num filled"
                )
    # Print detailed column comparison table
    print_dataset_column_table(column_info, list(combined_data.columns))

    logger.info("✓ Combined training data ready")
    logger.info(f"  Total samples: {len(combined_data):,}")
    logger.info(f"  Total columns: {len(combined_data.columns)}")
    logger.info(f"  First 5 columns: {combined_data.columns[:5].tolist()}")
    logger.info(f"  Datasets: {', '.join(dataset_names)}")

    if "p_num" in combined_data.columns or "id" in combined_data.columns:
        patient_col = "p_num" if "p_num" in combined_data.columns else "id"
        n_patients = len(combined_data[patient_col].unique())
        logger.info(f"  Total patients: {n_patients}")

    # Data quality checks for potential scaling issues
    logger.info(" ")
    logger.info("Data Quality Checks:")
    logger.info("-" * 80)

    issues_found = False
    for col in combined_data.columns:
        # Skip non-numeric columns
        if combined_data[col].dtype not in [
            "float64",
            "float32",
            "int64",
            "int32",
            "float16",
            "int16",
        ]:
            continue

        nan_count = combined_data[col].isna().sum()
        nan_pct = (nan_count / len(combined_data)) * 100

        # Check for NaN values
        if nan_count > 0:
            logger.warning(f"  ⚠ {col}: {nan_count:,} NaN values ({nan_pct:.2f}%)")
            issues_found = True

        # Check for zero variance (constant columns)
        non_nan_values = combined_data[col].dropna()
        if len(non_nan_values) > 0:
            std_val = non_nan_values.std()
            if std_val == 0 or (std_val is not None and abs(std_val) < 1e-10):
                unique_val = (
                    non_nan_values.iloc[0] if len(non_nan_values) > 0 else "N/A"
                )
                logger.warning(
                    f"  ⚠ {col}: Zero variance (constant value: {unique_val})"
                )
                issues_found = True

        # Check for infinite values
        if combined_data[col].dtype in ["float64", "float32", "float16"]:
            inf_count = combined_data[col].isin([float("inf"), float("-inf")]).sum()
            if inf_count > 0:
                logger.warning(f"  ⚠ {col}: {inf_count:,} infinite values")
                issues_found = True

    if not issues_found:
        logger.info("  ✓ No data quality issues detected in numeric columns")
    else:
        logger.warning(
            "  ⚠ Data quality issues detected - may cause scaling warnings during preprocessing"
        )

    logger.info("-" * 80)

    # Apply imputation to handle NaN values and prepare data for model
    logger.info(" ")
    logger.info("Applying data preprocessing:")
    logger.info("-" * 80)

    # Get numeric columns that need imputation
    numeric_cols = [
        col
        for col in combined_data.columns
        if combined_data[col].dtype
        in ["float64", "float32", "int64", "int32", "float16", "int16"]
    ]

    # Impute missing values
    logger.info("  Imputing missing values in numeric columns...")
    for col in numeric_cols:
        nan_before = combined_data[col].isna().sum()
        if nan_before > 0:
            combined_data = impute_missing_values(combined_data, columns=[col])
            nan_after = combined_data[col].isna().sum()
            logger.info(f"    • {col}: {nan_before:,} → {nan_after:,} NaN values")

    # Check for zero variance columns after imputation
    zero_variance_cols = []
    for col in numeric_cols:
        non_nan = combined_data[col].dropna()
        if len(non_nan) > 0 and non_nan.std() == 0:
            zero_variance_cols.append(col)

    if zero_variance_cols:
        logger.warning(f"  ⚠ Columns with zero variance detected: {zero_variance_cols}")
        logger.warning(
            "    These columns will be dropped as they provide no information"
        )
        combined_data = combined_data.drop(columns=zero_variance_cols)
        logger.info(f"    Dropped {len(zero_variance_cols)} zero-variance columns")

    logger.info("  ✓ Data preprocessing completed")
    logger.info(
        f"  Final shape: {combined_data.shape[0]:,} rows x {combined_data.shape[1]:,} columns"
    )
    logger.info(f"  With remaining columns: {combined_data.columns.tolist()}")
    logger.info(f"  Data example:\n{combined_data.head(5)}")
    logger.info("-" * 80)

    return combined_data


def step4_zero_shot_evaluation(
    model_type: str,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    batch_size: int = 2048,
    model_config_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Step 4: Zero-shot evaluation using pretrained model (no fine-tuning).

    This demonstrates the model's pretrained capabilities on glucose forecasting
    before any domain-specific fine-tuning. Uses the proper zero-shot configuration
    with freeze_backbone=True and num_epochs=0.

    Args:
        model_type: Type of model to use (ttm, chronos, moment)
        dataset_names: List of dataset names
        training_columns: Column names from training data
        config_dir: Holdout config directory
        output_dir: Output directory
        batch_size: Batch size for inference
        model_config_overrides: Optional dict of model-specific config from YAML

    Note: This step creates a temporary model just for zero-shot evaluation.
    Step 5 will create a fresh model for fine-tuning.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4: Zero-Shot Evaluation (Pretrained Model)")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU count: {gpu_info['gpu_count']}")

    # Single GPU (no distributed)
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    # Create zero-shot configuration using the factory
    config = ModelFactory.create_zero_shot_config(
        model_type=model_type,
        batch_size=batch_size,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
        extra_config=model_config_overrides,
    )

    logger.info("Zero-shot model config:")
    logger.info(f"  Model type: {config.model_type}")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Training mode: {config.training_mode}")
    logger.info(f"  Freeze backbone: {config.freeze_backbone}")
    logger.info(f"  Num epochs: {config.num_epochs}")

    # Create model using the factory
    model = ModelFactory.create_model(config, distributed_config=distributed_config)
    logger.info(f"✓ Pretrained {model_type.upper()} model loaded (zero-shot mode)")

    # Evaluate and plot for zero-shot phase
    _evaluate_and_plot(
        model=model,
        training_columns=training_columns,
        dataset_names=dataset_names,
        config_dir=config_dir,
        output_dir=output_dir,
        phase_name="0_zero_shot",
        zero_shot=True,
    )

    logger.info("✓ Zero-shot evaluation completed")
    # Note: We don't return the model - step5 will create a fresh one for training

    # Explicitly free GPU memory before step 5 creates a new model
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU memory cleared after zero-shot evaluation")
    except Exception:
        pass


def step5_train_model(
    model_type: str,
    combined_data: pd.DataFrame,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, GenericModelConfig, Dict, Path]:
    """Step 5: Fine-tune model on combined dataset.

    Creates a fresh model configured for fine-tuning (not zero-shot).

    Args:
        model_type: Type of model to use (ttm, chronos, moment)
        combined_data: Combined training DataFrame
        dataset_names: List of dataset names
        training_columns: Column names from training data
        config_dir: Holdout config directory
        output_dir: Output directory
        num_epochs: Number of training epochs (None = use YAML or default)
        batch_size: Batch size for training (None = use YAML or default)
        model_config_overrides: Optional dict of model-specific config from YAML

    Returns:
        tuple: (model, config, results, model_path) - Trained model, config, training results, and save path
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 5: Fine-tune Model")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info(
        f"Epochs: {num_epochs if num_epochs is not None else 'from YAML or default'}"
    )
    logger.info("=" * 80)

    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    # Create fine-tuning configuration using the factory
    config = ModelFactory.create_finetune_config(
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
        extra_config=model_config_overrides,
    )

    logger.info("Fine-tuning config:")
    logger.info(f"  Model type: {config.model_type}")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Training mode: {config.training_mode}")
    logger.info(f"  Freeze backbone: {config.freeze_backbone}")
    logger.info(f"  Num epochs: {config.num_epochs}")

    # Create fresh model for fine-tuning using the factory
    model = ModelFactory.create_model(config, distributed_config=distributed_config)
    logger.info(f"✓ Fresh {model_type.upper()} model created for fine-tuning")

    print(f"\n>>> Starting training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} epoch(s)...\n")
    logger.info(f"Training on combined datasets: {', '.join(dataset_names)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Train the model (fit() is implemented by each model type)
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print("\n>>> Training completed successfully\n")
        logger.info("✓ Training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save model checkpoint (save() is implemented by base class)
        model_path = Path(output_dir) / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

        # Evaluate and plot after training
        _evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="1_after_training",
        )

        return model, config, results, model_path

    except Exception as e:
        print(f"\n>>> ERROR: Training failed: {e}\n")
        logger.error(f"✗ Training failed: {e}")
        raise


def step6_load_checkpoint(
    model_type: str,
    model_path: Path,
    config: GenericModelConfig,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
) -> Optional[Any]:
    """Step 6: Load model from checkpoint and verify it works.

    This step demonstrates that the model can be saved and loaded correctly.

    Args:
        model_type: Type of model (ttm, chronos, moment)
        model_path: Path to the saved model checkpoint
        config: GenericModelConfig for loading the model
        training_columns: Column names from training data
        dataset_names: List of dataset names
        config_dir: Holdout config directory
        output_dir: Output directory

    Returns:
        Loaded model instance, or None if loading failed
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 6: Load Model from Checkpoint")
    logger.info("=" * 80)

    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"✗ Model file not found: {model_path}")
        return None
    else:
        logger.info(f"✓ Model file found: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        # Load using the class method
        # Create a temporary model via factory to access the correct class's load()
        model = ModelFactory.load_model(
            model_type=model_type,
            model_path=str(model_path),
            config=config,
        )
        logger.info(f"✓ Model loaded from: {model_path}")

        # Evaluate and plot after loading (to verify it works)
        _evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="2_after_loading",
        )

        return model

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        traceback.print_exc()
        return None


def step7_resume_training(
    model,  # BaseTimeSeriesFoundationModel or compatible
    combined_data: pd.DataFrame,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    num_epochs: Optional[int] = None,
) -> Tuple[Any, Dict, Path]:
    """Step 7: Resume training on loaded model for additional epochs.

    This demonstrates the ability to continue training from a checkpoint.
    The training_history attribute comes from the base class.

    Args:
        model: Loaded model instance
        combined_data: Training data
        dataset_names: List of dataset names
        training_columns: Column names from training data
        config_dir: Holdout config directory
        output_dir: Output directory
        num_epochs: Number of additional epochs

    Returns:
        tuple: (model, results, model_path) - Updated model, results, and save path
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 7: Resume Training on Loaded Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info(
        f"Additional epochs: {num_epochs if num_epochs is not None else 'from YAML or default'}"
    )
    logger.info("=" * 80)

    # Check if model has training history from previous training
    # training_history comes from the base class
    if hasattr(model, "training_history"):
        logger.info("✓ Model has training history from previous training")
        if isinstance(model.training_history, dict) and model.training_history:
            if "log_history" in model.training_history:
                logger.info(
                    f"  Log history entries: {len(model.training_history['log_history'])}"
                )
            if "best_metric" in model.training_history:
                logger.info(
                    f"  Best metric from previous training: {model.training_history['best_metric']}"
                )
    else:
        logger.warning("⚠ Model does not have training_history attribute")

    # Create output directory for resumed training
    resumed_output_dir = Path(output_dir) / "resumed_training"
    resumed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Resuming training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {resumed_output_dir}")
    epochs_display = num_epochs if num_epochs is not None else "configured"
    print(f">>> Training with {epochs_display} additional epoch(s)...\n")

    try:
        # Continue training (fit() is implemented by child class)
        results = model.fit(
            train_data=combined_data, output_dir=str(resumed_output_dir)
        )
        print("\n>>> Resumed training completed successfully\n")
        logger.info("✓ Resumed training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save the model after resumed training (save() is from base class)
        model_path = resumed_output_dir / "model.pt"
        model.save(str(model_path))
        logger.info(f"✓ Resumed model saved to: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

        # Evaluate and plot after resumed training
        _evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="3_after_resumed_training",
        )

        return model, results, model_path

    except Exception as e:
        print(f"\n>>> ERROR: Resumed training failed: {e}\n")
        logger.error(f"✗ Resumed training failed: {e}")
        raise


def step8_full_holdout_evaluation(
    model,  # BaseTimeSeriesFoundationModel or compatible
    dataset_names: list,
    config_dir: str,
) -> Dict[str, Any]:
    """Step 8: Full evaluation on holdout sets for all datasets.

    This performs the comprehensive evaluation using the model's evaluate()
    method on the complete holdout data for each dataset.

    Args:
        model: Trained model instance
        dataset_names: List of dataset names
        config_dir: Holdout config directory

    Returns:
        dict: Mapping of dataset names to evaluation results
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 8: Full Holdout Evaluation")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    registry = DatasetRegistry(holdout_config_dir=config_dir)
    all_results = {}

    for dataset_name in dataset_names:
        logger.info(f"\n--- Evaluating holdout for: {dataset_name} ---")

        # Load holdout data
        holdout_data = registry.load_holdout_data_only(dataset_name)
        logger.info(f"✓ Holdout data loaded: {len(holdout_data):,} samples")

        # Log dataset info
        if "p_num" in holdout_data.columns or "id" in holdout_data.columns:
            patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
            holdout_patients = holdout_data[patient_col].dropna().unique()
            logger.info(f"  Holdout patients: {len(holdout_patients)}")

        # Pass full holdout data to evaluate() - the model's _prepare_inference_data()
        # needs id/timestamp columns (p_num, datetime) for ForecastDFDataset.
        # Only exclude non-feature metadata columns like source_dataset.
        eval_data = holdout_data.drop(
            columns=["source_dataset"], errors="ignore"
        ).copy()

        logger.info(f"  Columns for evaluation: {list(eval_data.columns)}")
        logger.info(f"  Holdout data shape: {eval_data.shape}")

        # Evaluate using the evaluate() method
        try:
            logger.info("  Running evaluation on holdout set...")

            eval_results = model.evaluate(test_data=eval_data)

            logger.info(f"  ✓ Evaluation completed for {dataset_name}")
            logger.info("  Metrics:")

            # Log all metrics from evaluation
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    - {key}: {value:.6f}")
                else:
                    logger.info(f"    - {key}: {value}")

            all_results[dataset_name] = eval_results

        except Exception as e:
            logger.error(f"  ✗ Evaluation failed for {dataset_name}: {e}")
            traceback.print_exc()
            all_results[dataset_name] = None

    # Summary
    logger.info("=" * 80)
    logger.info("Full Holdout Evaluation Summary")
    logger.info("=" * 80)
    for dataset_name, results in all_results.items():
        if results is not None:
            # Find primary metric (usually MSE or loss)
            primary_metric = results.get("eval_loss", results.get("mse", "N/A"))
            if isinstance(primary_metric, float):
                logger.info(f"  {dataset_name}: eval_loss = {primary_metric:.6f}")
            else:
                logger.info(f"  {dataset_name}: {primary_metric}")
        else:
            logger.info(f"  {dataset_name}: FAILED")

    return all_results


# =============================================================================
# MAIN WORKFLOW
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end holdout workflow for time series foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
  1. Check holdout configs exist
  2. Validate holdout configs
  3. Load and combine training data
  4. Zero-shot evaluation (pretrained model, no fine-tuning)
  5. Fine-tune model for specified epochs
  6. Load model from checkpoint (verify save/load works)
  7. Resume training on loaded model
  8. Full holdout evaluation on all datasets

Supported Model Types:
  - ttm: IBM Granite TTM (TinyTimeMixer)
  - chronos: Amazon Chronos
  - moment: AutonLab MOMENT

Each evaluation phase (4, 5, 6, 7) generates predictions and plots
stored in separate subdirectories for comparison.
        """,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ttm",
        choices=["ttm", "chronos", "moment"],
        help="Type of model to use (default: ttm)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names to combine (e.g., lynch_2022 aleppo brown_2019)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Training output directory"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training steps (only run zero-shot and load existing model)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs per phase. Overrides YAML config value. "
        "Falls back to YAML num_epochs, then default (1) if not set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training and inference. Overrides YAML config value. "
        "Falls back to YAML batch_size, then default (2048) if not set.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model YAML config file (e.g., configs/models/ttm/default.yaml). "
        "Specifies model-specific parameters like input_features, scaler_type, "
        "split_config, etc. Explicit CLI args (--epochs, --batch-size) override YAML values.",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.output_dir = f"./trained_models/artifacts/_tsfm_testing/{timestamp}_{args.model_type}_holdout_workflow"

    # Load model config from YAML if provided
    model_config_overrides = None
    if args.model_config:
        model_config_overrides = load_model_config_from_yaml(args.model_config)

    logger.info("=" * 80)
    logger.info("🚀 GENERIC FORECASTER WORKFLOW DEMONSTRATION")
    logger.info("Start of: example_holdout_generic_workflow.py")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model_type.upper()}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model config: {args.model_config or 'None (using defaults)'}")
    logger.info(
        f"Epochs per phase: {args.epochs if args.epochs is not None else 'from YAML or default'}"
    )
    logger.info(f"Skip training: {args.skip_training}")
    logger.info("=" * 80)

    try:
        # Copy model config YAML to output artifacts for reproducibility
        if args.model_config:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            model_config_dest = output_path / "model_config.yaml"
            shutil.copy2(args.model_config, model_config_dest)
            logger.info(f"✓ Model config copied to: {model_config_dest}")
        # =====================================================================
        # STEP 1: Check/generate holdout configs
        # =====================================================================
        if not step1_generate_holdout_configs(
            args.config_dir, args.output_dir, args.datasets
        ):
            logger.error("Please generate holdout configs first")
            return

        # =====================================================================
        # STEP 2: Validate configuration for all datasets
        # =====================================================================
        if not step2_validate_holdout_configs(args.datasets, args.config_dir):
            logger.error("Configuration validation failed")
            return

        # =====================================================================
        # STEP 3: Load and combine training data
        # =====================================================================
        combined_train_data = step3_load_training_data(
            args.datasets, args.config_dir, args.output_dir
        )
        training_columns = list(combined_train_data.columns)

        # =====================================================================
        # STEP 4: Zero-shot evaluation (pretrained model, no fine-tuning)
        # =====================================================================
        step4_zero_shot_evaluation(
            model_type=args.model_type,
            dataset_names=args.datasets,
            training_columns=training_columns,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            model_config_overrides=model_config_overrides,
        )

        if args.skip_training:
            logger.info(" ")
            logger.info("=" * 80)
            logger.info("⏭️  SKIPPING TRAINING STEPS (--skip-training flag set)")
            logger.info("=" * 80)

            # Try to load existing model for evaluation
            model_path = Path(args.output_dir) / "model.pt"
            if model_path.exists():
                logger.info(f"Loading existing model from: {model_path}")

                # Create a config for loading (use YAML if provided)
                config = ModelFactory.create_finetune_config(
                    model_type=args.model_type,
                    extra_config=model_config_overrides,
                )

                model = step6_load_checkpoint(
                    model_type=args.model_type,
                    model_path=model_path,
                    config=config,
                    training_columns=training_columns,
                    dataset_names=args.datasets,
                    config_dir=args.config_dir,
                    output_dir=args.output_dir,
                )
                if model is None:
                    logger.error("Failed to load existing model")
                    return
            else:
                logger.info("No existing model found, skipping evaluation steps")
                return
        else:
            # =====================================================================
            # STEP 5: Fine-tune model for one epoch
            # =====================================================================
            model, config, train_results, model_path = step5_train_model(
                model_type=args.model_type,
                combined_data=combined_train_data,
                dataset_names=args.datasets,
                training_columns=training_columns,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                model_config_overrides=model_config_overrides,
            )

            # =====================================================================
            # STEP 6: Load model from checkpoint (verify save/load works)
            # =====================================================================
            model = step6_load_checkpoint(
                model_type=args.model_type,
                model_path=model_path,
                config=config,
                training_columns=training_columns,
                dataset_names=args.datasets,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
            )
            if model is None:
                logger.error("Failed to load model from checkpoint")
                return

            # =====================================================================
            # STEP 7: Resume training on loaded model
            # =====================================================================
            model, resume_results, resumed_model_path = step7_resume_training(
                model=model,
                combined_data=combined_train_data,
                dataset_names=args.datasets,
                training_columns=training_columns,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
            )

        # =====================================================================
        # STEP 8: Full holdout evaluation on all datasets
        # =====================================================================
        step8_full_holdout_evaluation(
            model=model,
            dataset_names=args.datasets,
            config_dir=args.config_dir,
        )

        # =====================================================================
        # WORKFLOW COMPLETE
        # =====================================================================
        logger.info("=" * 80)
        logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model type: {args.model_type.upper()}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Generated artifacts:")
        logger.info("  - predictions/0_zero_shot/       : Zero-shot predictions")
        if not args.skip_training:
            logger.info(
                "  - predictions/1_after_training/  : Post-training predictions"
            )
            logger.info("  - predictions/2_after_loading/   : Post-load predictions")
            logger.info(
                "  - predictions/3_after_resumed_training/ : Post-resume predictions"
            )
            logger.info("  - model.pt                       : Initial trained model")
            logger.info("  - resumed_training/model.pt      : Resumed training model")
        logger.info("  - forecasts/*/                   : Forecast plots per phase")
        logger.info("=" * 80)
        logger.info("End of: example_holdout_generic_workflow.py")

    except KeyboardInterrupt:
        logger.info("🛑 Workflow interrupted by user")
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
