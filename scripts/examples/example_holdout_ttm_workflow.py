#!/usr/bin/env python3
"""
End-to-end example: Holdout system with TTM training and evaluation.

This script demonstrates the complete TTMForecaster workflow including:
1. Generate holdout configurations
2. Validate configurations
3. Load and combine training data from multiple datasets
4. Zero-shot evaluation (pretrained model, no fine-tuning)
5. Fine-tune model for one epoch and evaluate
6. Load checkpointed model and evaluate (verify save/load works)
7. Resume training on loaded model and evaluate
8. Full holdout evaluation on all datasets

Usage:
    # Run with default settings (combine all 4 datasets, 5% holdout, 1 epoch)
    sbatch scripts/examples/run_holdout_ttm_workflow.sh

    # Run with specific datasets combined
    sbatch --export=DATASETS="lynch_2022 aleppo" scripts/examples/run_holdout_ttm_workflow.sh

    # Use different config directory
    sbatch --export=DATASETS="lynch_2022 brown_2019",CONFIG_DIR="configs/data/holdout" scripts/examples/run_holdout_ttm_workflow.sh

    # Customize number of epochs
    sbatch --export=DATASETS="aleppo brown_2019",EPOCHS=2 scripts/examples/run_holdout_ttm_workflow.sh

    # Direct python call (for testing)
    python scripts/examples/example_holdout_ttm_workflow.py --datasets lynch_2022 brown_2019 --config-dir configs/data/holdout_5pct --epochs 1
"""

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.dataset_combiner import (
    combine_datasets_for_training,
    print_dataset_column_table,
)
from src.data.preprocessing.imputation import impute_missing_values
from src.models.base import DistributedConfig, GPUManager
from src.models.ttm import TTMForecaster, TTMConfig
from src.models.ttm.config import create_ttm_zero_shot_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from data processing modules (but not versioning)
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)
logging.getLogger("src.models").setLevel(logging.WARNING)
logging.getLogger("src.utils").setLevel(logging.WARNING)

# =============================================================================
# HELPER FUNCTIONS: Prediction Generation and Plotting
# =============================================================================


def _generate_forecasts(
    model: TTMForecaster,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    zero_shot: bool = False,
):
    """Helper: Generate forecasts using the model and holdout data.

    Args:
        model: TTM forecaster (trained or pretrained)
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

            # Get first patient
            patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
            first_patient = holdout_data[patient_col].iloc[0]
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

            # Keep necessary columns for TTM preprocessing (p_num, datetime)
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

            import json

            with open(predictions_json, "w") as f:
                json.dump(predictions_data, f, indent=2)

            logger.info(f"    ✓ Predictions saved to: {predictions_json}")

        logger.info(f"  ✓ Forecast generation completed for phase: {phase_name}")
        return forecast_results

    except Exception as e:
        logger.error(f"  ✗ Failed to generate forecasts: {e}")
        import traceback

        traceback.print_exc()
        return None


def _plot_forecasts(
    forecast_results: dict,
    output_dir: str,
    phase_name: str,
):
    """Helper: Create plots and save forecast visualizations.

    Args:
        forecast_results: Dictionary from _generate_forecasts containing forecast data
        output_dir: Directory where plots will be saved
        phase_name: Identifier for this phase (e.g., "zero_shot", "after_training")
    """
    logger.info(f"  Plotting forecasts for phase: {phase_name}")

    if forecast_results is None:
        logger.error("  ✗ No forecast results to plot")
        return False

    try:
        import pandas as pd

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

            # Ensure predictions is 1D for plotting
            predictions = np.array(predictions).squeeze()
            logger.info(f"    Predictions shape for plotting: {predictions.shape}")
            logger.info(
                f"    Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]"
            )

            # Create plot
            plt.figure(figsize=(15, 6))

            # Plot historical data
            historical_time = np.arange(len(historical_glucose))
            plt.plot(
                historical_time,
                historical_glucose,
                "b-",
                label="Historical Data",
                linewidth=2,
            )

            # Plot actual future values
            actual_time = np.arange(
                len(historical_glucose), len(historical_glucose) + len(actual_glucose)
            )
            plt.plot(actual_time, actual_glucose, "g-", label="Actual", linewidth=2)

            # Plot forecast
            forecast_time = np.arange(
                len(historical_glucose), len(historical_glucose) + len(predictions)
            )
            plt.plot(forecast_time, predictions, "r--", label="Forecast", linewidth=2)

            # Add vertical line at forecast start
            plt.axvline(
                x=len(historical_glucose),
                color="gray",
                linestyle=":",
                linewidth=1.5,
                label="Forecast Start",
            )

            # Add reference lines for hypo/hyper thresholds (in mM)
            plt.axhline(
                y=3.9,
                color="orange",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Hypoglycemia (3.9 mM)",
            )
            plt.axhline(
                y=10.0,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Hyperglycemia (10.0 mM)",
            )

            # Labels and title
            plt.xlabel("Time Steps", fontsize=12)
            plt.ylabel("Blood Glucose (mM)", fontsize=12)
            plt.title(
                f"[{phase_name.upper()}] Blood Glucose Forecast - {dataset_name} "
                f"(Context: {context_length}, Patient: {patient_id})",
                fontsize=14,
            )
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = forecast_dir / f"{phase_name}_{dataset_name}_forecast.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"    ✓ Forecast plot saved to: {plot_path}")

            plt.close()

            # Save forecast data to CSV
            forecast_csv_path = forecast_dir / f"{phase_name}_{dataset_name}_data.csv"
            forecast_data_df = pd.DataFrame(
                {
                    "time_step": list(historical_time) + list(actual_time),
                    "historical": list(historical_glucose)
                    + [np.nan] * len(actual_glucose),
                    "actual": [np.nan] * len(historical_glucose) + list(actual_glucose),
                    "forecast": [np.nan] * len(historical_glucose) + list(predictions),
                }
            )
            forecast_data_df.to_csv(forecast_csv_path, index=False)
            logger.info(f"    ✓ Forecast data saved to: {forecast_csv_path}")

        logger.info(f"  ✓ Forecast plotting completed for phase: {phase_name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to plot forecasts: {e}")
        import traceback

        traceback.print_exc()
        return False


def _evaluate_and_plot(
    model: TTMForecaster,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
    phase_name: str,
    zero_shot: bool = False,
):
    """Helper: Generate forecasts and plots for a given phase.

    This is the main helper that combines forecast generation and plotting.
    Called after each major workflow phase (zero-shot, training, loading, etc.)

    Args:
        model: TTM forecaster
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
):
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


def step2_validate_holdout_configs(datasets: list, config_dir: str):
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


def step3_load_training_data(dataset_names: list, config_dir: str):
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
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
):
    """Step 4: Zero-shot evaluation using pretrained model (no fine-tuning).

    This demonstrates the TTM's pretrained capabilities on glucose forecasting
    before any domain-specific fine-tuning. Uses the proper zero-shot configuration
    with freeze_backbone=True and num_epochs=0.

    Note: This step creates a temporary model just for zero-shot evaluation.
    Step 5 will create a fresh model for fine-tuning.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4: Zero-Shot Evaluation (Pretrained Model)")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU count: {gpu_info['gpu_count']}")

    # Single GPU (no distributed)
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    # Use proper zero-shot configuration
    # This sets training_mode="zero_shot", freeze_backbone=True, num_epochs=0
    # Following tsfm_public pattern: TinyTimeMixerForPrediction.from_pretrained(...)
    config = create_ttm_zero_shot_config(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=2048,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
    )

    logger.info("Zero-shot model config:")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Training mode: {config.training_mode}")
    logger.info(f"  Freeze backbone: {config.freeze_backbone}")
    logger.info(f"  Num epochs: {config.num_epochs}")

    # Create model (loads pretrained weights with frozen backbone)
    model = TTMForecaster(config, distributed_config=distributed_config)
    logger.info("✓ Pretrained TTM model loaded (zero-shot mode)")

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


def step5_train_model(
    combined_data,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    num_epochs: int = 1,
):
    """Step 5: Fine-tune TTM model on combined dataset.

    Creates a fresh model configured for fine-tuning (not zero-shot).

    Args:
        combined_data: Combined training DataFrame
        dataset_names: List of dataset names
        training_columns: Column names from training data
        config_dir: Holdout config directory
        output_dir: Output directory
        num_epochs: Number of training epochs

    Returns:
        tuple: (model, config, results, model_path) - Trained model, config, training results, and save path
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 5: Fine-tune TTM Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    # Create a fresh model configured for fine-tuning (not zero-shot)
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=2048,
        num_epochs=num_epochs,
        training_mode="fine_tune",
        freeze_backbone=False,  # Trainable for fine-tuning
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
    )

    logger.info("Fine-tuning config:")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Training mode: {config.training_mode}")
    logger.info(f"  Freeze backbone: {config.freeze_backbone}")
    logger.info(f"  Num epochs: {config.num_epochs}")

    # Create fresh model for fine-tuning
    model = TTMForecaster(config, distributed_config=distributed_config)
    logger.info("✓ Fresh TTM model created for fine-tuning")

    print(f"\n>>> Starting training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} epoch(s)...\n")
    logger.info(f"Training on combined datasets: {', '.join(dataset_names)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Train the model
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print("\n>>> Training completed successfully\n")
        logger.info("✓ Training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save model checkpoint
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
    model_path: Path,
    config: TTMConfig,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
):
    """Step 6: Load model from checkpoint and verify it works.

    This step demonstrates that the model can be saved and loaded correctly.

    Args:
        model_path: Path to the saved model checkpoint
        config: TTMConfig for loading the model
        training_columns: Column names from training data
        dataset_names: List of dataset names
        config_dir: Holdout config directory
        output_dir: Output directory

    Returns:
        TTMForecaster: Loaded model
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
        model = TTMForecaster.load(str(model_path), config)
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
        import traceback

        traceback.print_exc()
        return None


def step7_resume_training(
    model: TTMForecaster,
    combined_data,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    num_epochs: int = 1,
):
    """Step 7: Resume training on loaded model for additional epochs.

    This demonstrates the ability to continue training from a checkpoint.

    Args:
        model: Loaded TTM model
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
    logger.info(f"Additional epochs: {num_epochs}")
    logger.info("=" * 80)

    # Check if model has training history from previous training
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
    print(f">>> Training with {num_epochs} additional epoch(s)...\n")

    try:
        # Continue training
        results = model.fit(
            train_data=combined_data, output_dir=str(resumed_output_dir)
        )
        print("\n>>> Resumed training completed successfully\n")
        logger.info("✓ Resumed training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save the model after resumed training
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
    model: TTMForecaster, dataset_names: list, config_dir: str
):
    """Step 8: Full evaluation on holdout sets for all datasets.

    This performs the comprehensive evaluation using Trainer.evaluate()
    on the complete holdout data for each dataset.

    Args:
        model: Trained TTM model
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
            holdout_patients = holdout_data[patient_col].unique()
            logger.info(f"  Holdout patients: {len(holdout_patients)}")

        # Prepare data for evaluation (remove non-numeric columns)
        exclude_cols = ["datetime", "p_num", "id", "source_dataset"]
        numeric_cols = [col for col in holdout_data.columns if col not in exclude_cols]
        holdout_data_numeric = holdout_data[numeric_cols].copy()

        logger.info(f"  Numeric columns for evaluation: {len(numeric_cols)}")
        logger.info(f"  Holdout data shape: {holdout_data_numeric.shape}")

        # Evaluate using the evaluate() method
        try:
            logger.info("  Running evaluation on holdout set...")

            eval_results = model.evaluate(test_data=holdout_data_numeric)

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
            import traceback

            traceback.print_exc()
            all_results[dataset_name] = None

    # Summary
    logger.info("\n" + "=" * 80)
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
        description="End-to-end holdout workflow demonstrating all TTMForecaster capabilities",
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

Each evaluation phase (4, 5, 6, 7) generates predictions and plots
stored in separate subdirectories for comparison.
        """,
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
        default=1,
        help="Number of training epochs per phase (default: 1)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.output_dir = (
            f"./trained_models/artifacts/_tsfm_testing/{timestamp}_holdout_workflow"
        )

    logger.info("=" * 80)
    logger.info("🚀 TTM FORECASTER COMPLETE WORKFLOW DEMONSTRATION")
    logger.info("Start of: example_holdout_ttm_workflow.py")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs per phase: {args.epochs}")
    logger.info(f"Skip training: {args.skip_training}")
    logger.info("=" * 80)

    try:
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
        combined_train_data = step3_load_training_data(args.datasets, args.config_dir)
        training_columns = list(combined_train_data.columns)

        # =====================================================================
        # STEP 4: Zero-shot evaluation (pretrained model, no fine-tuning)
        # =====================================================================
        step4_zero_shot_evaluation(
            dataset_names=args.datasets,
            training_columns=training_columns,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
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
                # Create a config for loading (same as fine-tuning config)
                # gpu_info = GPUManager.get_gpu_info()
                config = TTMConfig(
                    model_path="ibm-granite/granite-timeseries-ttm-r2",
                    context_length=512,
                    forecast_length=96,
                )
                model = step6_load_checkpoint(
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
                combined_data=combined_train_data,
                dataset_names=args.datasets,
                training_columns=training_columns,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
            )

            # =====================================================================
            # STEP 6: Load model from checkpoint (verify save/load works)
            # =====================================================================
            model = step6_load_checkpoint(
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
        logger.info("\n" + "=" * 80)
        logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Generated artifacts:")
        logger.info("  - predictions/zero_shot/          : Zero-shot predictions")
        if not args.skip_training:
            logger.info(
                "  - predictions/after_training/     : Post-training predictions"
            )
            logger.info("  - predictions/after_loading/      : Post-load predictions")
            logger.info(
                "  - predictions/after_resumed_training/ : Post-resume predictions"
            )
            logger.info("  - model.pt                         : Initial trained model")
            logger.info("  - resumed_training/model.pt        : Resumed training model")
        logger.info("  - forecasts/*/                     : Forecast plots per phase")
        logger.info("=" * 80)
        logger.info("End of: example_holdout_ttm_workflow.py")

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Workflow interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
