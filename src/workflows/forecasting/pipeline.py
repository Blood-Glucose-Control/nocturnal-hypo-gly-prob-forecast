#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""End-to-end forecasting workflow for multi-model training and evaluation.

The workflow executes a consistent seven-step sequence across supported model
families: data-split config checks, validation, training data assembly,
optional zero-shot evaluation, fine-tuning, checkpoint reload verification, and
resumed training.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.dataset_combiner import (
    combine_datasets_for_training,
    print_dataset_column_table,
)
from src.workflows.forecasting.evaluation import (
    evaluate_and_plot as phase_evaluate_and_plot,
)
from src.workflows.forecasting.modeling import (
    GenericModelConfig,
    ModelFactory,
    load_model_config_from_yaml as load_workflow_model_config_from_yaml,
)
from src.workflows.runtime.hardware import (
    clear_cuda_cache,
    get_gpu_info as runtime_get_gpu_info,
)

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
        logger.info("  Run: python scripts/data_processing/generate_holdout_configs.py")
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
                f"  Temporal holdout: {config.temporal_config.holdout_percentage * 100}%"
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
    dataset_names: list, config_dir: str, output_dir: Optional[str] = None
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

    # Impute missing values (sktime not available in all envs, e.g. chronos2)
    try:
        from src.data.preprocessing.imputation import impute_missing_values

        logger.info("  Imputing missing values in numeric columns...")
        for col in numeric_cols:
            nan_before = combined_data[col].isna().sum()
            if nan_before > 0:
                combined_data = impute_missing_values(combined_data, columns=[col])
                nan_after = combined_data[col].isna().sum()
                logger.info(f"    • {col}: {nan_before:,} → {nan_after:,} NaN values")
    except ImportError:
        logger.warning(
            "  sktime not installed — skipping imputation. "
            "Model must handle NaN values internally."
        )

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
) -> Any:
    """Step 4: Zero-shot evaluation using pretrained model (no fine-tuning).

    This demonstrates the model's pretrained capabilities on glucose forecasting
    before any domain-specific fine-tuning. Uses the proper zero-shot configuration
    with freeze_backbone=True and num_epochs=0.

    Args:
        model_type: Type of model to use (ttm, chronos, moment, timesfm)
        dataset_names: List of dataset names
        training_columns: Column names from training data
        config_dir: Holdout config directory
        output_dir: Output directory
        batch_size: Batch size for inference
        model_config_overrides: Optional dict of model-specific config from YAML

    Returns:
        The loaded zero-shot model instance. With --skip-training, this model
        is reused for step 8 evaluation. Otherwise, it is freed before step 5.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 4: Zero-Shot Evaluation (Pretrained Model)")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = runtime_get_gpu_info(logger)
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU count: {gpu_info['gpu_count']}")

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
    model = ModelFactory.create_model(config)

    # Check if model actually supports zero-shot prediction
    if not model.supports_zero_shot:
        logger.info(
            f"{model_type} does not support zero-shot (trains from scratch). "
            "Skipping zero-shot evaluation."
        )
        del model
        clear_cuda_cache(logger, context="after skipping zero-shot evaluation")
        return None

    logger.info(f"✓ Pretrained {model_type.upper()} model loaded (zero-shot mode)")

    # Evaluate and plot for zero-shot phase
    phase_evaluate_and_plot(
        model=model,
        training_columns=training_columns,
        dataset_names=dataset_names,
        config_dir=config_dir,
        output_dir=output_dir,
        phase_name="0_zero_shot",
        model_config_overrides=model_config_overrides,
    )

    logger.info("✓ Zero-shot evaluation completed")

    return model


def step5_train_model(
    model_type: str,
    combined_data: pd.DataFrame,
    dataset_names: list,
    training_columns: list,
    config_dir: str,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 2048,
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
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        model_config_overrides: Optional dict of model-specific config from YAML

    Returns:
        tuple: (model, config, results, model_path) - Trained model, config, training results, and save path
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 5: Fine-tune Model")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info("=" * 80)

    # GPU setup
    gpu_info = runtime_get_gpu_info(logger)
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
    model = ModelFactory.create_model(config)
    logger.info(f"✓ Fresh {model_type.upper()} model created for fine-tuning")

    print(f"\n>>> Starting training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} epoch(s)...\n")
    logger.info(f"Training on combined datasets: {', '.join(dataset_names)}")
    logger.info(f"Output directory: {output_dir}")

    # Filter training data to only model config columns if specified
    # This ensures the preprocessor only learns scalers for the features we'll use at inference
    train_data_for_model = combined_data
    if model_config_overrides:
        # Guard against YAML null values converting to None
        input_features = model_config_overrides.get("input_features") or []
        target_features = model_config_overrides.get("target_features") or []
        if input_features or target_features:
            required_cols = ["p_num", "id", "datetime"]
            model_cols = list(input_features) + list(target_features)
            all_cols = [
                col
                for col in model_cols + required_cols
                if col in combined_data.columns
            ]
            train_data_for_model = combined_data[all_cols].copy()
            logger.info(f"Filtered training data to model config columns: {model_cols}")
            logger.info(f"  Training data shape: {train_data_for_model.shape}")

    try:
        # Train the model (fit() is implemented by each model type)
        results = model.fit(train_data=train_data_for_model, output_dir=output_dir)
        print("\n>>> Training completed successfully\n")
        logger.info("✓ Training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save model checkpoint (save() is implemented by base class)
        model_path = Path(output_dir) / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

        # Evaluate and plot after training
        phase_evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="1_after_training",
            model_config_overrides=model_config_overrides,
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
    model_config_overrides: Optional[Dict[str, Any]] = None,
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
        model_config_overrides: Optional dict of model-specific config from YAML

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
        logger.info(f"  Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

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
        phase_evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="2_after_loading",
            model_config_overrides=model_config_overrides,
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
    num_epochs: int = 1,
    model_config_overrides: Optional[Dict[str, Any]] = None,
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
        model_config_overrides: Optional dict of model-specific config from YAML

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

    print(f">>> Resuming training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {resumed_output_dir}")
    print(f">>> Training with {num_epochs} additional epoch(s)...\n")

    # Filter training data to same columns used in initial training
    train_data_for_model = combined_data
    if model_config_overrides:
        # Guard against YAML null values converting to None
        input_features = model_config_overrides.get("input_features") or []
        target_features = model_config_overrides.get("target_features") or []
        if input_features or target_features:
            required_cols = ["p_num", "id", "datetime"]
            model_cols = list(input_features) + list(target_features)
            all_cols = [
                col
                for col in model_cols + required_cols
                if col in combined_data.columns
            ]
            train_data_for_model = combined_data[all_cols].copy()

    try:
        # Continue training (fit() is implemented by child class)
        results = model.fit(
            train_data=train_data_for_model, output_dir=str(resumed_output_dir)
        )
        print("\n>>> Resumed training completed successfully\n")
        logger.info("✓ Resumed training completed")
        logger.info(f"  Results: {list(results.keys())}")

        # Save the model after resumed training (save() is from base class)
        model_path = resumed_output_dir / "model.pt"
        model.save(str(model_path))
        logger.info(f"✓ Resumed model saved to: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

        # Evaluate and plot after resumed training
        phase_evaluate_and_plot(
            model=model,
            training_columns=training_columns,
            dataset_names=dataset_names,
            config_dir=config_dir,
            output_dir=output_dir,
            phase_name="3_after_resumed_training",
            model_config_overrides=model_config_overrides,
        )

        return model, results, model_path

    except Exception as e:
        print(f"\n>>> ERROR: Resumed training failed: {e}\n")
        logger.error(f"✗ Resumed training failed: {e}")
        raise


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


@dataclass
class ForecastingWorkflowRequest:
    """Programmatic request surface for running the forecasting workflow."""

    model_type: str
    datasets: list[str]
    config_dir: str = "configs/data/holdout_10pct"
    output_dir: Optional[str] = None
    skip_training: bool = False
    skip_steps: list[int] = field(default_factory=list)
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    model_config: Optional[str] = None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end forecasting workflow for time series foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
  1. Check holdout configs exist
  2. Validate holdout configs
  3. Load and combine training data
  4. Zero-shot evaluation (pretrained model, no fine-tuning)
  5. Train model for specified epochs
  6. Load model from checkpoint (verify save/load works)
  7. Resume training on loaded model

Supported Model Types:
  - ttm: IBM Granite TTM (TinyTimeMixer)
  - chronos: Amazon Chronos
  - moment: AutonLab MOMENT
  - timesfm: Google TimesFM 2.0 (500M)
  - timegrad: TimeGrad (GRU + diffusion, trains from scratch)
  - tide: TiDE (Time-series Dense Encoder, trains from scratch via AutoGluon)

Step 4 is auto-skipped for from-scratch models like timegrad, tide.

Each evaluation phase (4, 5, 6, 7) generates predictions and plots
stored in separate subdirectories for comparison.
        """,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ttm",
        choices=[
            "ttm",
            "chronos",
            "chronos2",
            "moment",
            "timesfm",
            "timegrad",
            "tide",
            "toto",
            "moirai",
            "naive_baseline",
            "statistical",
            "deepar",
            "patchtst",
            "tft",
        ],
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
        default="configs/data/holdout_10pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Training output directory"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training steps 5-7 (equivalent to --skip-steps 5 6 7)",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        nargs="+",
        default=[],
        help="Step numbers to skip (e.g., --skip-steps 4 7 to skip zero-shot and resume)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs per phase (default: from YAML config, or 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training and inference (default: from YAML config, or 2048)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to YAML model configuration file (e.g., configs/models/ttm/fine_tune.yaml)",
    )

    return parser


def run_with_args(args: argparse.Namespace) -> int:
    entrypoint_name = Path(sys.argv[0]).as_posix()

    # Load model config from YAML if provided
    model_config_overrides = None
    if args.model_config:
        model_config_overrides = load_workflow_model_config_from_yaml(args.model_config)

    # Build the set of steps to skip
    skip_steps: set[int] = set(args.skip_steps)
    if args.skip_training:
        skip_steps.update({5, 6, 7})
    # Auto-skip zero-shot if YAML explicitly sets training_mode=from_scratch.
    # The model-level check (model.supports_zero_shot) happens inside step 4
    # after the model is created — no hardcoded model type list needed here.
    resolved_training_mode = (model_config_overrides or {}).get(
        "training_mode", "fine_tune"
    )
    if resolved_training_mode == "from_scratch" and 4 not in skip_steps:
        logger.info(
            f"Auto-skipping step 4 (zero-shot): "
            f"{args.model_type} is configured with training_mode=from_scratch"
        )
        skip_steps.add(4)

    # Set output directory
    # Always create a unique timestamped RID subdirectory so that:
    #   - repeated runs never clobber each other, and
    #   - the artifacts root (e.g. trained_models/artifacts/ttm) is not polluted.
    _now = datetime.now()
    _ts_short = _now.strftime("%Y-%m-%d_%H:%M")
    _ts_long = _now.strftime("%Y%m%d_%H%M%S")
    _pid = os.getpid()
    _run_subdir = f"{_ts_short}_RID{_ts_long}_{_pid}_forecasting_workflow"
    if args.output_dir is None:
        args.output_dir = f"./trained_models/artifacts/_tsfm_testing/{_run_subdir}"
    else:
        args.output_dir = str(Path(args.output_dir) / _run_subdir)

    logger.info("=" * 80)
    logger.info("GENERIC FORECASTER WORKFLOW DEMONSTRATION")
    logger.info(f"Start of: {entrypoint_name}")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model_type.upper()}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model config: {args.model_config or 'None (using defaults)'}")
    logger.info(f"Epochs per phase: {args.epochs}")
    logger.info(f"Skip steps: {sorted(skip_steps) if skip_steps else 'none'}")
    logger.info("=" * 80)

    # Copy model config to output directory for reproducibility
    if args.model_config:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model_config, output_path / "model_config.yaml")
        logger.info(f"Copied model config to: {output_path / 'model_config.yaml'}")

    try:
        # =====================================================================
        # STEP 1: Check/generate holdout configs
        # =====================================================================
        if 1 not in skip_steps:
            if not step1_generate_holdout_configs(
                args.config_dir, args.output_dir, args.datasets
            ):
                logger.error("Please generate holdout configs first")
                return 1
        else:
            logger.info("Skipping step 1 (holdout config check)")

        # =====================================================================
        # STEP 2: Validate configuration for all datasets
        # =====================================================================
        if 2 not in skip_steps:
            if not step2_validate_holdout_configs(args.datasets, args.config_dir):
                logger.error("Configuration validation failed")
                return 1
        else:
            logger.info("Skipping step 2 (config validation)")

        # =====================================================================
        # STEP 3: Load and combine training data
        # =====================================================================
        if 3 not in skip_steps:
            combined_train_data = step3_load_training_data(
                args.datasets, args.config_dir, args.output_dir
            )
            training_columns = list(combined_train_data.columns)
        else:
            logger.info("Skipping step 3 (load training data)")
            combined_train_data = None
            training_columns = []

        # =====================================================================
        # STEP 4: Zero-shot evaluation (pretrained model, no fine-tuning)
        # =====================================================================
        if 4 not in skip_steps:
            zero_shot_model = step4_zero_shot_evaluation(
                model_type=args.model_type,
                dataset_names=args.datasets,
                training_columns=training_columns,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                model_config_overrides=model_config_overrides,
            )
        else:
            zero_shot_model = None
            logger.info("Skipping step 4 (zero-shot evaluation)")

        # =====================================================================
        # STEP 5: Train model
        # =====================================================================
        model = None
        config = None
        model_path = None

        if 5 not in skip_steps:
            # Free zero-shot model GPU memory before step 5 creates a new model
            if zero_shot_model is not None:
                del zero_shot_model
                clear_cuda_cache(logger, context="after zero-shot evaluation")

            if combined_train_data is None:
                logger.error("Step 5 requires training data (step 3)")
                return 1
            model, config, _, model_path = step5_train_model(
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
        else:
            logger.info("Skipping step 5 (training)")

        # =====================================================================
        # STEP 6: Load model from checkpoint (verify save/load works)
        # =====================================================================
        if 6 not in skip_steps:
            # If step 5 was skipped, try loading from default path
            if model_path is None:
                model_path = Path(args.output_dir) / "model.pt"
            if config is None:
                config = ModelFactory.create_finetune_config(
                    model_type=args.model_type,
                    extra_config=model_config_overrides,
                )

            if not Path(model_path).exists():
                logger.warning(
                    f"No model checkpoint found at {model_path}, skipping step 6"
                )
            else:
                model = step6_load_checkpoint(
                    model_type=args.model_type,
                    model_path=model_path,
                    config=config,
                    training_columns=training_columns,
                    dataset_names=args.datasets,
                    config_dir=args.config_dir,
                    output_dir=args.output_dir,
                    model_config_overrides=model_config_overrides,
                )
                if model is None:
                    logger.error("Failed to load model from checkpoint")
                    return 1
        else:
            logger.info("Skipping step 6 (load checkpoint)")

        # =====================================================================
        # STEP 7: Resume training on loaded model
        # =====================================================================
        if 7 not in skip_steps:
            if model is None or combined_train_data is None:
                logger.warning(
                    "Step 7 requires a loaded model (step 6) and training data "
                    "(step 3), skipping"
                )
            else:
                model, _, _ = step7_resume_training(
                    model=model,
                    combined_data=combined_train_data,
                    dataset_names=args.datasets,
                    training_columns=training_columns,
                    config_dir=args.config_dir,
                    output_dir=args.output_dir,
                    num_epochs=args.epochs,
                    model_config_overrides=model_config_overrides,
                )
        else:
            logger.info("Skipping step 7 (resume training)")

        # =====================================================================
        # WORKFLOW COMPLETE
        # =====================================================================
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model type: {args.model_type.upper()}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Steps skipped: {sorted(skip_steps) if skip_steps else 'none'}")
        logger.info("Generated artifacts:")
        if 4 not in skip_steps:
            logger.info("  - predictions/0_zero_shot/       : Zero-shot predictions")
        if 5 not in skip_steps:
            logger.info(
                "  - predictions/1_after_training/  : Post-training predictions"
            )
            logger.info("  - model.pt                       : Trained model")
        if 6 not in skip_steps:
            logger.info("  - predictions/2_after_loading/   : Post-load predictions")
        if 7 not in skip_steps:
            logger.info(
                "  - predictions/3_after_resumed_training/ : Post-resume predictions"
            )
            logger.info("  - resumed_training/model.pt      : Resumed training model")
        logger.info("  - forecasts/*/                   : Forecast plots per phase")
        logger.info("=" * 80)
        logger.info(f"End of: {entrypoint_name}")
        return 0

    except KeyboardInterrupt:
        logger.info("\n\nWorkflow interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\nWorkflow failed: {e}")
        traceback.print_exc()
        return 1


def run_workflow(request: ForecastingWorkflowRequest) -> int:
    args = argparse.Namespace(
        model_type=request.model_type,
        datasets=request.datasets,
        config_dir=request.config_dir,
        output_dir=request.output_dir,
        skip_training=request.skip_training,
        skip_steps=request.skip_steps,
        epochs=request.epochs,
        batch_size=request.batch_size,
        model_config=request.model_config,
    )
    return run_with_args(args)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    return run_with_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
