#!/usr/bin/env python3
"""
End-to-end example: Holdout system with TTM training and evaluation.

This script demonstrates the complete workflow:
1. Generate holdout configurations
2. Validate configurations
3. Load and combine training data from multiple datasets
4. Train TTM model on combined data
5. Save model
6. Load model
7. Evaluate on holdout sets per dataset

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from data processing modules (but not versioning)
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)
logging.getLogger("src.models").setLevel(logging.WARNING)
logging.getLogger("src.utils").setLevel(logging.WARNING)


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


def step4_train_ttm_model(
    combined_data, dataset_names: list, output_dir: str, num_epochs: int = 1
):
    """Step 4: Train TTM model on combined dataset."""
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4: Train TTM Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU count: {gpu_info['gpu_count']}")

    # Single GPU training (no distributed)
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    # Create logging directory path with dataset info
    # Note: output_dir already includes job ID from the shell script
    # logging_dir = Path(output_dir) / f"hf_model_training_logs"
    # logging_dir.mkdir(parents=True, exist_ok=True)
    # logger.info(f"Creating HF logging directory: \n\t {logging_dir}")

    # TTM configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=2048,
        learning_rate=1e-4,
        num_epochs=num_epochs,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
    )

    logger.info("Model config:")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    # logger.info(f"  Logging directory: {logging_dir}")

    # Create model
    model = TTMForecaster(config, distributed_config=distributed_config)
    logger.info("✓ TTM model created")

    # Train
    print(f"\n>>> Starting training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} epoch(s)...\n")
    logger.info(f"Training on combined datasets: {', '.join(dataset_names)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Pass the combined DataFrame directly instead of dataset name
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print("\n>>> Training completed successfully on combined datasets\n")
        logger.info("✓ Training completed")
        logger.info(f"  Results: {list(results.keys())}")
        return model, results
    except Exception as e:
        print(f"\n>>> ERROR: Training failed: {e}\n")
        logger.error(f"✗ Training failed: {e}")
        raise


def step4b_generate_forecasts(
    model: TTMForecaster,
    training_columns: list,
    dataset_names: list,
    config_dir: str,
    output_dir: str,
):
    """Step 4b: Generate forecasts using the trained model and holdout data.

    Args:
        model: Trained TTM forecaster
        training_columns: List of column names used during training
        dataset_names: List of dataset names to generate forecasts for
        config_dir: Directory containing holdout configurations
        output_dir: Directory where prediction CSV files will be saved

    Returns:
        Dict[str, Dict]: Dictionary mapping dataset names to forecast results containing:
            - predictions: Model predictions array
            - historical_glucose: Historical glucose values
            - actual_glucose: Actual future glucose values
            - patient_id: ID of the patient used for forecasting
            - context_length: Length of context window
            - forecast_length: Length of forecast horizon
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4b: Generate Forecasts")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    try:
        context_length = model.config.context_length
        forecast_length = model.config.forecast_length
        registry = DatasetRegistry(holdout_config_dir=config_dir)

        # Create predictions output directory
        predictions_dir = Path(output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Predictions output directory: {predictions_dir}")

        forecast_results = {}

        # Process each dataset's first holdout patient
        for dataset_name in dataset_names:
            logger.info(f"--- Generating forecast for dataset: {dataset_name} ---")

            # Load holdout data
            holdout_data = registry.load_holdout_data_only(dataset_name)

            # Get first patient
            patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
            first_patient = holdout_data[patient_col].iloc[0]
            patient_data = holdout_data[holdout_data[patient_col] == first_patient]

            logger.info(
                f"First holdout datframe: {first_patient}"
            )  # Will be a temporal holdout slice
            logger.info(f"Patient data shape: {patient_data.shape}")
            logger.info(f"Patient data preview:\n{patient_data.head()}")

            forecast_cols = [
                col for col in training_columns if col in patient_data.columns
            ]
            # Slice to get context + forecast length
            total_length = context_length + forecast_length
            forecast_data = patient_data.iloc[:total_length][forecast_cols].copy()

            logger.info(f"Forecast data shape: {forecast_data.shape}")
            logger.info(f"Forecast data preview:\n{forecast_data.head()}")

            # Keep necessary columns for TTM preprocessing (p_num, datetime)
            # Only remove source_dataset if present
            exclude_cols = ["source_dataset"]
            forecast_cols_for_model = [
                col for col in forecast_data.columns if col not in exclude_cols
            ]
            forecast_data_for_model = forecast_data[forecast_cols_for_model].copy()

            logger.info(
                f"Forecast data for model shape: {forecast_data_for_model.shape}"
            )
            logger.info(f"Columns for model: {forecast_cols_for_model}")

            # Generate predictions
            predictions = model.predict(forecast_data_for_model)

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

            logger.info(f"✓ Generated forecast for {dataset_name}")
            logger.info(f"  Predictions shape: {predictions.shape}")
            logger.info(f"  Predictions preview: {predictions[:5]}")
            logger.info(f"  Historical data points: {len(historical_glucose)}")
            logger.info(f"  Actual future data points: {len(actual_glucose)}")

            # Save predictions to JSON for quick inspection (handles multi-dimensional data better)
            predictions_json = (
                predictions_dir
                / f"predictions_{dataset_name}_patient{first_patient}.json"
            )

            # Prepare data for JSON serialization
            predictions_data = {
                "dataset": dataset_name,
                "patient_id": str(first_patient),
                "predictions_shape": list(predictions.shape),
                "predictions": predictions.tolist(),  # Convert numpy to list for JSON
                "forecast_length": forecast_length,
                "context_length": context_length,
            }

            # Add datetime info if available
            if forecast_datetimes is not None:
                predictions_data["forecast_datetimes"] = [
                    str(dt) for dt in forecast_datetimes
                ]

            # Save as JSON
            import json

            with open(predictions_json, "w") as f:
                json.dump(predictions_data, f, indent=2)

            logger.info(f"  ✓ Predictions saved to: {predictions_json}")

        logger.info("\n✓ Forecast generation completed successfully")
        return forecast_results

    except Exception as e:
        logger.error(f"✗ Failed to generate forecasts: {e}")
        import traceback

        traceback.print_exc()
        return None


def step4c_plot_forecasts(
    forecast_results: dict,
    output_dir: str,
):
    """Step 4c: Create plots and save forecast visualizations.

    Args:
        forecast_results: Dictionary from step4b_generate_forecasts containing forecast data
        output_dir: Directory where plots and CSV files will be saved
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4c: Plot Forecasts")
    logger.info("=" * 80)

    if forecast_results is None:
        logger.error("✗ No forecast results to plot")
        return False

    try:
        import pandas as pd

        # Create forecast output directory
        forecast_dir = Path(output_dir) / "forecasts"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Forecast output directory: {forecast_dir}")

        # Plot each dataset's forecast
        for dataset_name, results in forecast_results.items():
            logger.info(f"\n--- Plotting forecast for dataset: {dataset_name} ---")

            predictions = results["predictions"]
            historical_glucose = results["historical_glucose"]
            actual_glucose = results["actual_glucose"]
            patient_id = results["patient_id"]
            context_length = results["context_length"]

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
                f"Blood Glucose Forecast - {dataset_name} Context Length {context_length}, (Patient {patient_id})",
                fontsize=14,
            )
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = forecast_dir / f"forecast_plot_{dataset_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"✓ Forecast plot saved to: {plot_path}")

            plt.close()

            # Save forecast data to CSV
            forecast_csv_path = forecast_dir / f"forecast_data_{dataset_name}.csv"
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
            logger.info(f"✓ Forecast data saved to: {forecast_csv_path}")

        logger.info("\n✓ Forecast plotting completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to plot forecasts: {e}")
        import traceback

        traceback.print_exc()
        return False


def step5_save(model: TTMForecaster, save_path: Path):
    """Step 5: Save trained model."""
    logger.info("=" * 80)
    logger.info("STEP 5: Save Trained Model")
    logger.info("=" * 80)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(save_path))
        logger.info(f"✓ Model saved to: {save_path}")
        logger.info(f"  Size: {save_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save model: {e}")
        return False


def step6_load(load_path: str, config: TTMConfig):
    """Step 6: Load trained model."""
    logger.info("=" * 80)
    logger.info("STEP 6: Load Trained Model")
    logger.info("=" * 80)

    load_path = Path(load_path)

    if not load_path.exists():
        logger.error(f"✗ Model file not found: {load_path}")
        return None
    else:
        logger.info(f"✓ Model file found: {load_path}")

    try:
        # Load using the class method
        model = TTMForecaster.load(str(load_path), config)
        logger.info(f"✓ Model loaded from: {load_path}")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return None


def step6b_continue_training(
    model: TTMForecaster,
    combined_data,
    dataset_names: list,
    output_dir: str,
    num_epochs: int = 1,
):
    """Step 6b: Continue training on loaded model for additional epochs."""
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 6b: Continue Training Loaded Model")
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

    print(f"\n>>> Continuing training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} additional epoch(s)...\n")

    try:
        # Continue training - the model should resume from its current state
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print("\n>>> Continued training completed successfully\n")
        logger.info("✓ Continued training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Check training history after continued training
        if hasattr(model, "training_history"):
            if isinstance(model.training_history, dict) and model.training_history:
                if "log_history" in model.training_history:
                    logger.info(
                        f"  Updated log history entries: {len(model.training_history['log_history'])}"
                    )
                if "best_metric" in model.training_history:
                    logger.info(
                        f"  Best metric after continued training: {model.training_history['best_metric']}"
                    )

        return model, results
    except Exception as e:
        print(f"\n>>> ERROR: Continued training failed: {e}\n")
        logger.error(f"✗ Continued training failed: {e}")
        raise


def step7_evaluate_on_holdout(model: TTMForecaster, dataset_name: str, config_dir: str):
    """Step 7: Evaluate model on holdout set."""
    logger.info("=" * 80)
    logger.info(f"### STEP 7: Evaluate on Holdout Set for {dataset_name}")
    logger.info("=" * 80)
    registry = DatasetRegistry(holdout_config_dir=config_dir)

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

    # Evaluate using the new evaluate() method
    try:
        logger.info("Running evaluation on holdout set using Trainer.evaluate()...")

        # Call the evaluate method with holdout data
        eval_results = model.evaluate(test_data=holdout_data_numeric)

        logger.info("✓ Evaluation completed")
        logger.info("  Metrics:")

        # Log all metrics from evaluation
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"    - {key}: {value:.6f}")
            else:
                logger.info(f"    - {key}: {value}")

        return eval_results
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end holdout workflow with TTM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Skip training (use existing model)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.output_dir = f"./trained_models/artifacts/_tsfm_testing/{timestamp}_default_dir_holdout_workflow"

    model_path = Path(args.output_dir) / "model.pt"

    logger.info("=" * 80)
    logger.info("🚀 HOLDOUT SYSTEM WORKFLOW WITH TTM")
    logger.info("Start of: example_holdout_ttm_workflow.py")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 80)

    try:
        # Step 1: Check/generate holdout configs
        if not step1_generate_holdout_configs(
            args.config_dir, args.output_dir, args.datasets
        ):
            logger.error("Please generate holdout configs first")
            return

        # Step 2: Validate configuration for all datasets
        if not step2_validate_holdout_configs(args.datasets, args.config_dir):
            logger.error("Configuration validation failed")
            return

        # Step 3: Load and combine training data
        combined_train_data = step3_load_training_data(args.datasets, args.config_dir)

        if not args.skip_training:
            # Step 4: Train model on combined data
            model, results = step4_train_ttm_model(
                combined_train_data,
                args.datasets,
                args.output_dir,
                num_epochs=args.epochs,
            )
            logger.info(f"Plotting with columns: {list(combined_train_data.columns)}")
            # Step 4b: Generate forecasts
            forecast_results = step4b_generate_forecasts(
                model,
                list(combined_train_data.columns),
                args.datasets,
                args.config_dir,
                args.output_dir,
            )

            # Step 4c: Plot forecasts
            if forecast_results is not None:
                step4c_plot_forecasts(forecast_results, args.output_dir)

            # Step 5: Save model
            if not step5_save(model, str(model_path)):
                logger.error("Failed to save model")
                return
        else:
            # Step 4: Skip training (use existing model)
            logger.info("=" * 80)
            logger.info("STEP 4: Train TTM Model")
            logger.info("⏭️  Skipping training (using existing model)")
            logger.info("=" * 80)
            # Step 5: Skip saving model
            logger.info("=" * 80)
            logger.info("STEP 5: Save Trained Model")
            logger.info("⏭️  Skipping training (No trained model to save)")
            logger.info("=" * 80)

        # Step 6: Load model
        # Recreate config for loading
        gpu_info = GPUManager.get_gpu_info()
        config = TTMConfig(
            model_path="ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            forecast_length=96,
            batch_size=2048,
            learning_rate=1e-4,
            num_epochs=args.epochs,  # Use same epochs config
            use_cpu=not gpu_info["gpu_available"],
            fp16=gpu_info["gpu_available"],
        )

        model = step6_load(str(model_path), config)
        if model is None:
            logger.error("Failed to load model")
            return

        # Step 6b: Continue training for another epoch (to test save/load/resume)
        if not args.skip_training:
            logger.info(" ")
            logger.info("=" * 80)
            logger.info("🔄 Testing Save/Load/Resume: Continue training loaded model")
            logger.info("=" * 80)

            # Create a new output directory for continued training
            continued_output_dir = Path(args.output_dir) / "continued_training"
            continued_output_dir.mkdir(parents=True, exist_ok=True)

            model, continued_results = step6b_continue_training(
                model,
                combined_train_data,
                args.datasets,
                str(continued_output_dir),
                num_epochs=args.epochs,
            )

            # Save the model again after continued training
            continued_model_path = continued_output_dir / "model.pt"
            if not step5_save(model, str(continued_model_path)):
                logger.error("Failed to save continued training model")
                # Don't return - still evaluate
            else:
                logger.info(
                    f"✓ Continued training model saved to: {continued_model_path}"
                )

        # Step 7: Evaluate on holdout for each dataset
        for dataset_name in args.datasets:
            logger.info(f"\nEvaluating on holdout set for: {dataset_name}")
            step7_evaluate_on_holdout(model, dataset_name, args.config_dir)

        logger.info("\n" + "=" * 80)
        logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Training outputs in: {args.output_dir}")
        logger.info("End of : example_holdout_ttm_workflow.py")

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Workflow interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
