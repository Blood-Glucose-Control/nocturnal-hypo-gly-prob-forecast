"""
TTM (TinyTimeMixer) configuration classes.

This module provides configuration classes specific to TTM models,
extending the base model configuration with TTM-specific parameters.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.models.base import ModelConfig, TrainingStrategy
from tsfm_public.toolkit.time_series_preprocessor import ScalerType


@dataclass
class TTMTrainingConfig:
    """TTM-specific training configuration."""

    # TTM specific training parameters
    freeze_backbone: bool = False
    use_tracking_callback: bool = True
    find_optimal_lr: bool = False

    # Custom TTM loss functions
    loss_function: str = "mse"  # "mse", "mae", "huber", "custom_ttm"

    # TTM preprocessing
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    imputation_strategy: str = "mean"  # "mean", "median", "forward_fill"


@dataclass
class TTMDataConfig:
    """TTM-specific data configuration."""

    # Data features
    input_features: List[str] = None
    target_features: List[str] = None

    # Data splitting
    split_config: Dict[str, float] = None

    # Data preprocessing
    num_input_channels: int = 5
    num_output_channels: int = 1
    prediction_filter_length: Optional[int] = None

    def __post_init__(self):
        if self.input_features is None:
            self.input_features = [
                "cob",
                "carb_availability",
                "insulin_availability",
                "iob",
                "steps",
            ]

        if self.target_features is None:
            self.target_features = ["bg_mM"]

        if self.split_config is None:
            self.split_config = {"train": 0.7, "val": 0.2, "test": 0.1}


class TTMConfig(ModelConfig):
    """Extended configuration class for TTM-specific parameters."""

    def __init__(self, **kwargs):
        # Extract TTM-specific parameters before calling parent
        ttm_specific_params = {
            "scaler_type",
            "imputation_strategy",
            "input_features",
            "target_features",
            "split_config",
            "num_input_channels",
            "num_output_channels",
            "prediction_filter_length",
            "resolution_min",
            "use_tracking_callback",
            "find_optimal_lr",
        }

        # Filter out TTM-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ttm_specific_params}

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set TTM-specific defaults
        self.model_type = "ttm"
        self.training_strategy = TrainingStrategy.TRANSFORMERS
        # model_path is now handled by the parent class

        # TTM Training Configuration
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.use_tracking_callback = kwargs.get("use_tracking_callback", True)
        self.find_optimal_lr = kwargs.get("find_optimal_lr", False)

        # TTM Data Configuration
        self.scaler_type = kwargs.get("scaler_type", "standard")
        self.imputation_strategy = kwargs.get("imputation_strategy", "mean")

        self.input_features = kwargs.get(
            "input_features",
            ["cob", "carb_availability", "insulin_availability", "iob", "steps"],
        )
        self.target_features = kwargs.get("target_features", ["bg_mM"])

        self.split_config = kwargs.get(
            "split_config", {"train": 0.9, "val": 0.05, "test": 0.05}
        )
        self.fewshot_percent = kwargs.get("fewshot_percent", 5)

        # TTM Model Architecture
        self.num_input_channels = kwargs.get("num_input_channels", 5)
        self.num_output_channels = kwargs.get("num_output_channels", 1)
        self.prediction_filter_length = kwargs.get("prediction_filter_length", None)

        # Data resolution (specific to glucose data)
        self.resolution_min = kwargs.get("resolution_min", 5)

    def get_scaler_type(self) -> ScalerType:
        """Convert string scaler type to ScalerType enum."""
        scaler_mapping = {
            "standard": ScalerType.STANDARD,
            "minmax": ScalerType.MINMAX,
            "robust": ScalerType.ROBUST,
        }
        return scaler_mapping.get(self.scaler_type, ScalerType.STANDARD)

    def to_training_config(self) -> TTMTrainingConfig:
        """Convert to TTM-specific training config."""
        return TTMTrainingConfig(
            freeze_backbone=self.freeze_backbone,
            use_tracking_callback=self.use_tracking_callback,
            find_optimal_lr=self.find_optimal_lr,
            loss_function=self.loss_function,
            scaler_type=self.scaler_type,
            imputation_strategy=self.imputation_strategy,
        )

    def to_data_config(self) -> TTMDataConfig:
        """Convert to TTM-specific data config."""
        return TTMDataConfig(
            input_features=self.input_features,
            target_features=self.target_features,
            split_config=self.split_config,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            prediction_filter_length=self.prediction_filter_length,
        )

    def get_column_specifiers(self, data_columns: List[str]) -> Dict[str, List[str]]:
        """
        Create column specifiers for TTM data preprocessing.

        Args:
            data_columns: Available columns in the dataset

        Returns:
            Dictionary with id_columns, target_columns, observable_columns
        """
        # Filter available columns
        available_input_features = [
            col for col in self.input_features if col in data_columns
        ]
        available_target_features = [
            col for col in self.target_features if col in data_columns
        ]

        # Identify ID column (usually patient ID)
        id_columns = []
        for col in data_columns:
            if "patient" in col.lower() or "id" in col.lower():
                id_columns.append(col)
                break

        # If no ID column found, use index
        if not id_columns and "index" in data_columns:
            id_columns = ["index"]

        return {
            "id_columns": id_columns,
            "target_columns": available_target_features,
            "observable_columns": available_input_features,
        }

    def validate_config(self) -> bool:
        """Validate TTM configuration parameters."""
        errors = []

        # Check required parameters
        if not self.model_path:
            errors.append("model_path is required for TTM")

        if self.context_length <= 0:
            errors.append("context_length must be positive")

        if self.forecast_length <= 0:
            errors.append("forecast_length must be positive")

        if not self.input_features:
            errors.append("input_features cannot be empty")

        if not self.target_features:
            errors.append("target_features cannot be empty")

        # Check data split configuration
        split_sum = sum(self.split_config.values())
        if abs(split_sum - 1.0) > 1e-6:
            errors.append(f"Data split ratios must sum to 1.0, got {split_sum}")

        # Check scaler type
        valid_scalers = ["standard", "minmax", "robust"]
        if self.scaler_type not in valid_scalers:
            errors.append(f"scaler_type must be one of {valid_scalers}")

        if errors:
            error_msg = "TTM configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ValueError(error_msg)

        return True


def create_default_ttm_config(**overrides) -> TTMConfig:
    """
    Create a TTM configuration with sensible defaults.

    Args:
        **overrides: Configuration parameters to override

    Returns:
        TTMConfig instance with defaults applied
    """
    defaults = {
        # Model configuration
        "model_path": "ibm-granite/granite-timeseries-ttm-r2",
        "context_length": 512,
        "forecast_length": 96,
        # Training configuration
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        # TTM-specific
        "scaler_type": "standard",
        "imputation_strategy": "mean",
        "freeze_backbone": False,
        "use_tracking_callback": True,
        # Data configuration
        "input_features": [
            "cob",
            "carb_availability",
            "insulin_availability",
            "iob",
            "steps",
        ],
        "target_features": ["bg_mM"],
        "resolution_min": 5,
    }

    # Apply overrides
    defaults.update(overrides)

    # Create and validate config
    config = TTMConfig(**defaults)
    config.validate_config()

    return config


def create_ttm_fine_tuning_config(**overrides) -> TTMConfig:
    """Create TTM config optimized for fine-tuning."""
    fine_tuning_defaults = {
        "fit_strategy": "fine_tune",
        "freeze_backbone": False,
        "learning_rate": 1e-5,  # Lower LR for fine-tuning
        "num_epochs": 5,  # Fewer epochs for fine-tuning
        "warmup_steps": 500,  # Fewer warmup steps
    }

    return create_default_ttm_config(**fine_tuning_defaults, **overrides)


def create_ttm_zero_shot_config(**overrides) -> TTMConfig:
    """Create TTM config for zero-shot evaluation."""
    zero_shot_defaults = {
        "fit_strategy": "zero_shot",
        "freeze_backbone": True,
        "num_epochs": 0,  # No training
    }

    return create_default_ttm_config(**zero_shot_defaults, **overrides)
