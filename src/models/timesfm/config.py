"""Configuration for TimesFM model (HuggingFace Transformers)."""

from dataclasses import dataclass, field
from typing import Optional, List

from src.models.base import ModelConfig


@dataclass
class TimesFMConfig(ModelConfig):
    """Configuration for TimesFM model (HuggingFace implementation).

    Attributes:
        checkpoint_path: HF model ID or local path.
        context_length: Input context window length (default: 512).
        forecast_length: Alias for horizon_length (workflow compatibility).
        horizon_length: Forecast horizon (default: 96).
        batch_size: Per-device training batch size (default: 32).
        freq_type: Frequency encoding (0=5-min CGM, 1=hourly, 2=weekly+).
        torch_dtype: Model weight dtype ('bfloat16', 'float16', 'float32').
        gradient_accumulation_steps: Accumulate gradients over N steps.
        val_patient_ratio: Fraction of patients held out for validation (default: 0.2).
    """

    model_name: str = "timesfm"
    model_type: str = "timesfm"

    # Model path/checkpoint (aliases for compatibility)
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Sequence lengths (forecast_length aliases horizon_length for workflow compatibility)
    context_length: int = 512
    forecast_length: int = 96
    horizon_length: int = 96

    batch_size: int = 32
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    device: str = "auto"
    use_cpu: bool = False

    # Training parameters
    output_dir: Optional[str] = None
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    freq_type: int = 0
    val_patient_ratio: float = 0.2
    use_lora: bool = False

    # HF-specific
    torch_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 4

    # Data sampling
    window_stride: Optional[int] = None

    # Gap handling (applied before windowing)
    imputation_threshold_mins: int = 45

    def __post_init__(self):
        """Post-initialization to handle defaults, aliases, and validation."""
        # Handle None values from generic workflow
        if self.batch_size is None:
            self.batch_size = 32
        if self.horizon_length is None:
            self.horizon_length = self.forecast_length
        if self.forecast_length is None:
            self.forecast_length = self.horizon_length

        # Synchronize aliases
        if self.model_path and not self.checkpoint_path:
            self.checkpoint_path = self.model_path
        elif self.checkpoint_path and not self.model_path:
            self.model_path = self.checkpoint_path

        if self.forecast_length != self.horizon_length:
            self.horizon_length = self.forecast_length

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

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError(
                f"All quantiles must be between 0 and 1, got {self.quantiles}"
            )

        if self.torch_dtype not in ["bfloat16", "float16", "float32"]:
            raise ValueError(
                f"torch_dtype must be one of ['bfloat16', 'float16', 'float32'], "
                f"got {self.torch_dtype}"
            )
