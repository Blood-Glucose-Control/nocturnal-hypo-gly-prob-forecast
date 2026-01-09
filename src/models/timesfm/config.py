"""Configuration for TimesFM model.

This module contains the configuration class for the TimesFM model,
which wraps Google's TimesFM foundation model for time series forecasting.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from src.models.base import ModelConfig


@dataclass
class TimesFMConfig(ModelConfig):
    """Configuration for TimesFM model.

    TimesFM is a pretrained time series foundation model developed by Google Research.
    It uses a decoder-only transformer architecture trained on a large corpus of
    time series data.

    Attributes:
        model_name: Name of the model (default: "timesfm")
        model_type: Type of model architecture (default: "timesfm")
        model_path: Alias for checkpoint_path for consistency with BaseTSFM
        checkpoint_path: Path to pretrained checkpoint. If None, will download from HF
        context_length: Length of input context window (default: 512)
        forecast_length: Alias for horizon_length for consistency
        horizon_length: Length of forecast horizon (default: 128)
        input_patch_len: Length of input patches (default: 32)
        output_patch_len: Length of output patches (default: 128)
        num_layers: Number of transformer layers (default: 20)
        model_dims: Model dimension size (default: 1280)
        backend: Backend to use ('cpu', 'gpu', 'tpu') (default: 'cpu')
        per_core_batch_size: Batch size per device (default: 32)
        batch_size: Alias for per_core_batch_size
        quantiles: List of quantiles to predict for probabilistic forecasting
        use_cached_model: Whether to use cached model weights (default: True)
        device: Device to run model on ('cpu' or 'cuda')
        use_cpu: Whether to force CPU usage
        output_dir: Directory for saving outputs
        num_epochs: Number of training epochs (for adapter training)
        learning_rate: Learning rate (for adapter training)
        use_lora: Whether to use LoRA (reserved for future adapter support)
    """

    model_name: str = "timesfm"
    model_type: str = "timesfm"

    # Model path/checkpoint (both aliases for compatibility)
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Sequence lengths (with aliases for compatibility)
    context_length: int = 512
    forecast_length: int = 128  # Alias for horizon_length
    horizon_length: int = 128

    # Model architecture parameters
    input_patch_len: int = 32
    output_patch_len: int = 128
    num_layers: int = 20
    model_dims: int = 1280

    # Inference parameters
    backend: str = "cpu"  # Options: 'cpu', 'gpu', 'tpu'
    per_core_batch_size: int = 32
    batch_size: int = 32  # Alias for per_core_batch_size
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # System parameters
    use_cached_model: bool = True
    device: str = "cpu"
    use_cpu: bool = False

    # Training parameters (for future adapter support)
    output_dir: Optional[str] = None
    num_epochs: int = 10
    learning_rate: float = 1e-4
    use_lora: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        # Synchronize aliases
        if self.model_path and not self.checkpoint_path:
            self.checkpoint_path = self.model_path
        elif self.checkpoint_path and not self.model_path:
            self.model_path = self.checkpoint_path

        if self.forecast_length != self.horizon_length:
            # If they differ, use forecast_length as the source of truth
            self.horizon_length = self.forecast_length  # TODO: The config synchronization logic contains a potential issue. When forecast_length and horizon_length differ, forecast_length is used as the source of truth. However, if both are explicitly set to different values by the user, this will silently override horizon_length which could lead to unexpected behavior. Consider raising a warning or error when both are set to different values.

        if self.batch_size != self.per_core_batch_size:
            # If they differ, use batch_size as the source of truth
            self.per_core_batch_size = self.batch_size  # TODO: The config synchronization logic contains a potential issue. When forecast_length and horizon_length differ, forecast_length is used as the source of truth. However, if both are explicitly set to different values by the user, this will silently override horizon_length which could lead to unexpected behavior. Consider raising a warning or error when both are set to different values.

        # Set use_cpu based on device
        if self.device == "cpu":
            self.use_cpu = True

        # Validation
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )

        if self.horizon_length <= 0:
            raise ValueError(
                f"horizon_length must be positive, got {self.horizon_length}"
            )

        if self.backend not in ["cpu", "gpu", "tpu"]:
            raise ValueError(
                f"backend must be one of ['cpu', 'gpu', 'tpu'], got {self.backend}"
            )

        if self.per_core_batch_size <= 0:
            raise ValueError(
                f"per_core_batch_size must be positive, got {self.per_core_batch_size}"
            )

        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError(
                f"All quantiles must be between 0 and 1, got {self.quantiles}"
            )
