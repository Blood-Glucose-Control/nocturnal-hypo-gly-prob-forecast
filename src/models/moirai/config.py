"""
Moirai configuration classes.

This module provides configuration classes specific to Moirai models,
extending the base model configuration with Moirai-specific parameters.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class MoiraiTrainingConfig:
    """Moirai-specific training configuration.

    Contains training parameters specific to Moirai models that are not part
    of the base ModelConfig.

    Attributes:
        freeze_backbone: Whether to freeze the pre-trained Moirai backbone weights.
        use_tracking_callback: Whether to use experiment tracking callbacks.
        find_optimal_lr: Whether to run learning rate finder before training.
        loss_function: Loss function for training. Options: "mse", "mae",
            "huber", or "custom".
        scaler_type: Data normalization method. Options: "standard", "minmax",
            or "robust".
        imputation_strategy: Strategy for handling missing values. Options:
            "mean", "median", or "forward_fill".
    """

    # Moirai specific training parameters
    freeze_backbone: bool = False
    use_tracking_callback: bool = True
    find_optimal_lr: bool = False

    # Custom Moirai loss functions
    loss_function: str = "mse"  # "mse", "mae", "huber", "custom"

    # Moirai preprocessing
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    imputation_strategy: str = "mean"  # "mean", "median", "forward_fill"


@dataclass
class MoiraiDataConfig:
    """Moirai-specific data configuration.

    Contains data preprocessing and feature configuration specific to Moirai
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


class MoiraiConfig(ModelConfig):
    """Extended configuration class for Moirai-specific parameters.

    Inherits from ModelConfig and adds Moirai-specific attributes for training,
    data preprocessing, and model architecture configuration.

    Attributes:
        model_type: Always "moirai" for this configuration.
        model_path: HuggingFace model identifier or local path.
        context_length: Input sequence length (lookback window).
        forecast_length: Output prediction horizon.
        patch_size: Size of patches for Moirai's patching mechanism.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        training_config: Moirai-specific training parameters.
        data_config: Moirai-specific data configuration.
    """

    def __init__(self, **kwargs):
        """Initialize Moirai configuration.

        Args:
            **kwargs: Configuration parameters including:
                - model_path (str): Model identifier or path
                - context_length (int): Input sequence length
                - forecast_length (int): Output prediction horizon
                - patch_size (int): Patch size for patching
                - d_model (int): Model dimension
                - n_heads (int): Number of attention heads
                - n_layers (int): Number of transformer layers
                - dropout (float): Dropout rate
                - learning_rate (float): Learning rate
                - batch_size (int): Training batch size
                - num_epochs (int): Number of training epochs
                - use_lora (bool): Whether to use LoRA adaptation
        """
        # Extract Moirai-specific parameters before calling parent
        moirai_specific_params = {
            "patch_size",
            "d_model",
            "n_heads",
            "n_layers",
            "dropout",
            "training_config",
            "data_config",
        }

        # Filter out Moirai-specific params from kwargs for parent class
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in moirai_specific_params
        }

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Moirai-specific parameters
        self.model_type = "moirai"

        # Moirai architecture specifics
        self.patch_size = kwargs.get("patch_size", 16)
        self.d_model = kwargs.get("d_model", 768)
        self.n_heads = kwargs.get("n_heads", 12)
        self.n_layers = kwargs.get("n_layers", 12)
        self.dropout = kwargs.get("dropout", 0.1)

        # Initialize Moirai-specific configs
        self.training_config = kwargs.get("training_config", MoiraiTrainingConfig())
        self.data_config = kwargs.get("data_config", MoiraiDataConfig())

    def supports_lora(self) -> bool:
        """Check if Moirai supports LoRA fine-tuning.

        Returns:
            True, as Moirai is transformer-based and supports LoRA.
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
                "patch_size": self.patch_size,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            }
        )
        return base_dict


def create_default_moirai_config(**kwargs) -> MoiraiConfig:
    """Create a default Moirai configuration for zero-shot inference.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MoiraiConfig instance with default settings.
    """
    defaults = {
        "model_path": "Salesforce/moirai-1.0-R-small",
        "context_length": 512,
        "forecast_length": 96,
        "patch_size": 16,
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
    }
    defaults.update(kwargs)
    return MoiraiConfig(**defaults)


def create_moirai_fine_tuning_config(**kwargs) -> MoiraiConfig:
    """Create a Moirai configuration optimized for fine-tuning.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MoiraiConfig instance with fine-tuning settings.
    """
    defaults = {
        "model_path": "Salesforce/moirai-1.0-R-small",
        "context_length": 512,
        "forecast_length": 96,
        "training_strategy": TrainingBackend.FINE_TUNE,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 20,
    }
    defaults.update(kwargs)
    config = MoiraiConfig(**defaults)
    config.training_config.freeze_backbone = False
    return config


def create_moirai_zero_shot_config(**kwargs) -> MoiraiConfig:
    """Create a Moirai configuration for zero-shot inference.

    Args:
        **kwargs: Override default parameters.

    Returns:
        MoiraiConfig instance with zero-shot settings.
    """
    defaults = {
        "model_path": "Salesforce/moirai-1.0-R-small",
        "context_length": 512,
        "forecast_length": 96,
        "training_strategy": TrainingBackend.ZERO_SHOT,
        "batch_size": 64,
    }
    defaults.update(kwargs)
    config = MoiraiConfig(**defaults)
    config.training_config.freeze_backbone = True
    return config
