"""
Toto configuration classes.

This module provides configuration classes specific to Toto models,
extending the base model configuration with Toto-specific parameters.
"""

from typing import Dict, List
from dataclasses import dataclass

from src.models.base import ModelConfig, TrainingStrategy


@dataclass
class TotoTrainingConfig:
    """Toto-specific training configuration.

    Contains training parameters specific to Toto models that are not part
    of the base ModelConfig.

    Attributes:
        freeze_backbone: Whether to freeze the pre-trained Toto backbone weights.
        use_nll_loss: Whether to use negative log-likelihood loss (recommended for
            probabilistic outputs).
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        mse_weight: Weight for MSE loss component in composite loss (0.0 = pure NLL).
    """

    freeze_backbone: bool = False
    use_nll_loss: bool = True
    gradient_accumulation_steps: int = 1
    mse_weight: float = 0.1


@dataclass
class TotoDataConfig:
    """Toto-specific data configuration.

    Contains data preprocessing and feature configuration specific to Toto
    models, particularly for blood glucose forecasting applications.

    Attributes:
        input_features: List of input feature column names (e.g., ["bg_mM"]).
        target_feature: Target column name to predict (e.g., "bg_mM").
        timestamp_column: Name of the timestamp column.
        patient_id_column: Name of the patient ID column.
        split_config: Dictionary defining train/val/test split ratios.
    """

    input_features: List[str] | None = None
    target_feature: str = "bg_mM"
    timestamp_column: str = "datetime"
    patient_id_column: str = "p_num"
    split_config: Dict[str, float] | None = None

    def __post_init__(self):
        if self.input_features is None:
            self.input_features = ["bg_mM"]

        if self.split_config is None:
            self.split_config = {"train": 0.7, "val": 0.2, "test": 0.1}


class TotoConfig(ModelConfig):
    """Extended configuration class for Toto-specific parameters.

    Inherits from ModelConfig and adds Toto-specific attributes for training,
    data preprocessing, and model architecture configuration.

    Attributes:
        model_type: Always "toto" for this configuration.
        model_path: HuggingFace model path (default: Datadog/Toto-Open-Base-1.0).
        patch_size: Size of the input patches (default: 8, from pretrained model).
        stride: Stride for patch embedding (default: 8, from pretrained model).
        freeze_backbone: Whether to freeze pre-trained weights.
        use_nll_loss: Use NLL loss for probabilistic outputs.
        input_features: List of input feature column names.
        target_feature: Target column name to predict.
        split_config: Train/val/test split ratios dictionary.
        gradient_accumulation_steps: Steps to accumulate gradients.

    Example:
        >>> config = TotoConfig(
        ...     model_path="Datadog/Toto-Open-Base-1.0",
        ...     context_length=1024,
        ...     forecast_length=72,
        ...     batch_size=32,
        ... )
    """

    def __init__(self, **kwargs):
        # Extract Toto-specific parameters before calling parent
        toto_specific_params = {
            "patch_size",
            "stride",
            "input_features",
            "target_feature",
            "timestamp_column",
            "patient_id_column",
            "split_config",
            "use_nll_loss",
            "gradient_accumulation_steps",
            "mse_weight",
        }

        # Filter out Toto-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in toto_specific_params}

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Toto-specific defaults
        self.model_type = "toto"
        self.training_strategy = TrainingStrategy.TRANSFORMERS

        # Default model path if not provided
        if self.model_path is None:
            self.model_path = "Datadog/Toto-Open-Base-1.0"

        # Toto architecture parameters (from pretrained model)
        # Datadog/Toto-Open-Base-1.0 uses patch_size=64 and stride=64
        self.patch_size = kwargs.get("patch_size", 64)
        self.stride = kwargs.get("stride", 64)

        # Toto Training Configuration
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.use_nll_loss = kwargs.get("use_nll_loss", True)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.mse_weight = kwargs.get(
            "mse_weight", 0.1
        )  # Weight for MSE in composite loss

        # Toto Data Configuration
        self.input_features = kwargs.get("input_features", ["bg_mM"])
        self.target_feature = kwargs.get("target_feature", "bg_mM")
        self.timestamp_column = kwargs.get("timestamp_column", "datetime")
        self.patient_id_column = kwargs.get("patient_id_column", "p_num")

        self.split_config = kwargs.get(
            "split_config", {"train": 0.7, "val": 0.2, "test": 0.1}
        )

    def to_training_config(self) -> TotoTrainingConfig:
        """Convert to a TotoTrainingConfig instance.

        Returns:
            TotoTrainingConfig: Dataclass containing Toto training parameters.
        """
        return TotoTrainingConfig(
            freeze_backbone=self.freeze_backbone,
            use_nll_loss=self.use_nll_loss,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mse_weight=self.mse_weight,
        )

    def to_data_config(self) -> TotoDataConfig:
        """Convert to a TotoDataConfig instance.

        Returns:
            TotoDataConfig: Dataclass containing Toto data configuration.
        """
        return TotoDataConfig(
            input_features=self.input_features,
            target_feature=self.target_feature,
            timestamp_column=self.timestamp_column,
            patient_id_column=self.patient_id_column,
            split_config=self.split_config,
        )

    def validate_config(self) -> bool:
        """Validate Toto configuration parameters.

        Returns:
            bool: True if configuration is valid.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        errors = []

        # Check required parameters
        if not self.model_path:
            errors.append("model_path is required for Toto")

        if self.context_length <= 0:
            errors.append("context_length must be positive")

        if self.forecast_length <= 0:
            errors.append("forecast_length must be positive")

        if not self.input_features:
            errors.append("input_features cannot be empty")

        # Check data split configuration
        split_sum = sum(self.split_config.values())
        if abs(split_sum - 1.0) > 1e-6:
            errors.append(f"Data split ratios must sum to 1.0, got {split_sum}")

        # Check patch alignment - REQUIRED for Toto
        # Toto requires the total sequence length (context + forecast) to be divisible by patch_size
        total_length = self.context_length + self.forecast_length
        if total_length % self.patch_size != 0:
            errors.append(
                f"Total sequence length (context_length + forecast_length = "
                f"{self.context_length} + {self.forecast_length} = {total_length}) "
                f"must be divisible by patch_size ({self.patch_size}). "
                f"Adjust forecast_length to {self.forecast_length - (total_length % self.patch_size)} "
                f"or {self.forecast_length + (self.patch_size - (total_length % self.patch_size))}"
            )

        if errors:
            error_msg = "Toto configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ValueError(error_msg)

        return True


def create_default_toto_config(**overrides) -> TotoConfig:
    """Create a Toto configuration with sensible defaults for blood glucose forecasting.

    Args:
        **overrides: Configuration parameters to override

    Returns:
        TotoConfig instance with defaults applied
    """
    defaults = {
        # Model configuration
        "model_path": "Datadog/Toto-Open-Base-1.0",
        "context_length": 1024,  # ~85 hours at 5-min resolution (16 patches of 64)
        "forecast_length": 64,  # ~5.3 hours at 5-min resolution (1 patch of 64)
        # Total: 1024 + 64 = 1088 timesteps = 17 patches (divisible by patch_size=64)
        # Training configuration
        "batch_size": 32,
        "learning_rate": 1e-5,
        "num_epochs": 10,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        # Toto-specific (patch_size and stride default to 64 in __init__)
        "freeze_backbone": False,
        "use_nll_loss": True,
        # Data configuration
        "input_features": ["bg_mM"],
        "target_feature": "bg_mM",
    }

    # Apply overrides
    defaults.update(overrides)

    return TotoConfig(**defaults)


def create_toto_fine_tuning_config(**overrides) -> TotoConfig:
    """Create a TotoConfig optimized for fine-tuning.

    Uses lower learning rate suitable for fine-tuning a pre-trained model.

    Args:
        **overrides: Configuration parameters to override.

    Returns:
        TotoConfig: Configured for fine-tuning.
    """
    fine_tuning_defaults = {
        "fit_strategy": "fine_tune",
        "freeze_backbone": False,
        "learning_rate": 1e-5,
        "num_epochs": 10,
        "warmup_steps": 500,
    }

    return create_default_toto_config(**fine_tuning_defaults, **overrides)


def create_toto_zero_shot_config(**overrides) -> TotoConfig:
    """Create a TotoConfig for zero-shot evaluation.

    Args:
        **overrides: Configuration parameters to override.

    Returns:
        TotoConfig: Configured for zero-shot inference.
    """
    zero_shot_defaults = {
        "fit_strategy": "zero_shot",
        "freeze_backbone": True,
        "num_epochs": 0,
    }

    return create_default_toto_config(**zero_shot_defaults, **overrides)
