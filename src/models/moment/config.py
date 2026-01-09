"""
Moment configuration classes.

This module provides configuration classes specific to Moment models,
extending the base model configuration with Moment-specific parameters.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.models.base import ModelConfig, TrainingStrategy


@dataclass
class MomentTrainingConfig:
    """Moment-specific training configuration.

    Contains training parameters specific to Moment models that are not part
    of the base ModelConfig.

    Attributes:
        freeze_backbone: Whether to freeze the pre-trained Moment backbone weights.
        use_tracking_callback: Whether to use experiment tracking callbacks.
        find_optimal_lr: Whether to run learning rate finder before training.
        loss_function: Loss function for training. Options: "mse", "mae",
            "huber", or "custom".
        scaler_type: Data normalization method. Options: "standard", "minmax",
            or "robust".
        imputation_strategy: Strategy for handling missing values. Options:
            "mean", "median", or "forward_fill".
    """

    # Moment specific training parameters
    freeze_backbone: bool = False
    use_tracking_callback: bool = True
    find_optimal_lr: bool = False

    # Custom Moment loss functions
    loss_function: str = "mse"  # "mse", "mae", "huber", "custom"

    # Moment preprocessing
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    imputation_strategy: str = "mean"  # "mean", "median", "forward_fill"


@dataclass
class MomentDataConfig:
    """Moment-specific data configuration.

    Contains data preprocessing and feature configuration specific to Moment
    models, particularly for blood glucose forecasting applications.

    Attributes:
        input_features: List of input feature column names (e.g., ["cob", "iob"]).
        target_features: List of target column names to predict (e.g., ["bg_mM"]).
        split_config: Dictionary defining train/val/test split ratios.
            Must sum to 1.0. Example: {"train": 0.7, "val": 0.2, "test": 0.1}.
        num_input_channels: Number of input feature channels.
        num_output_channels: Number of output prediction channels.
        prediction_filter_length: Optional filter length for predictions.
            If None, uses full forecast_length.
    """

    # Data features
    input_features: List[str] | None = None
    target_features: List[str] | None = None

    # Data splitting
    split_config: Dict[str, float] | None = None
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


class MomentConfig(ModelConfig):
    """Extended configuration class for Moment-specific parameters.

    Inherits from ModelConfig and adds Moment-specific attributes for training,
    data preprocessing, and model architecture configuration.

    Attributes:
        model_type: Always "moment" for this configuration.
        model_path: HuggingFace model identifier or local path.
        context_length: Input sequence length (lookback window).
        forecast_length: Output prediction horizon.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        mask_ratio: Ratio of time steps to mask during pre-training.
        training_config: Moment-specific training parameters.
        data_config: Moment-specific data configuration.
    """

    def __init__(self, **kwargs):
        """Initialize Moment configuration.

        Args:
            **kwargs: Configuration parameters including:
                - model_path (str): Model identifier or path
                - context_length (int): Input sequence length
                - forecast_length (int): Output prediction horizon
                - d_model (int): Model dimension
                - n_heads (int): Number of attention heads
                - n_layers (int): Number of transformer layers
                - dropout (float): Dropout rate
                - mask_ratio (float): Masking ratio for pre-training
                - learning_rate (float): Learning rate
                - batch_size (int): Training batch size
                - num_epochs (int): Number of training epochs
                - use_lora (bool): Whether to use LoRA adaptation
        """
        # Extract Moment-specific parameters before calling parent
        moment_specific_params = {
            "d_model",
            "n_heads",
            "n_layers",
            "dropout",
            "mask_ratio",
            "training_config",
            "data_config",
        }

        # Filter out Moment-specific params from kwargs for parent class
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in moment_specific_params
        }

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Moment-specific parameters
        self.model_type = "moment"

        # Moment architecture specifics
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.n_layers = kwargs.get("n_layers", 6)
        self.dropout = kwargs.get("dropout", 0.1)
        self.mask_ratio = kwargs.get("mask_ratio", 0.15)

        # Initialize Moment-specific configs
        self.training_config = kwargs.get("training_config", MomentTrainingConfig())
        self.data_config = kwargs.get("data_config", MomentDataConfig())

    def supports_lora(self) -> bool:
        """Check if Moment supports LoRA fine-tuning.

        Returns:
            True, as Moment is transformer-based and supports LoRA.
        """
        return True

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "mask_ratio": self.mask_ratio,
            }
        )
        return base_dict


def create_default_moment_config(**kwargs) -> MomentConfig:
    """Create a default Moment configuration for zero-shot inference.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MomentConfig instance with default settings.
    """
    defaults = {
        "model_path": "AutonLab/MOMENT-1-large",
        "context_length": 512,
        "forecast_length": 96,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
    }
    defaults.update(kwargs)
    return MomentConfig(**defaults)


def create_moment_fine_tuning_config(**kwargs) -> MomentConfig:
    """Create a Moment configuration optimized for fine-tuning.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MomentConfig instance with fine-tuning settings.
    """
    defaults = {
        "model_path": "AutonLab/MOMENT-1-large",
        "context_length": 512,
        "forecast_length": 96,
        "training_strategy": TrainingStrategy.FINE_TUNE,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 20,
    }
    defaults.update(kwargs)
    config = MomentConfig(**defaults)
    config.training_config.freeze_backbone = False
    return config


def create_moment_zero_shot_config(**kwargs) -> MomentConfig:
    """Create a Moment configuration for zero-shot inference.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MomentConfig instance with zero-shot settings.
    """
    defaults = {
        "model_path": "AutonLab/MOMENT-1-large",
        "context_length": 512,
        "forecast_length": 96,
        "training_strategy": TrainingStrategy.ZERO_SHOT,
        "batch_size": 64,
    }
    defaults.update(kwargs)
    config = MomentConfig(**defaults)
    config.training_config.freeze_backbone = True
    return config
