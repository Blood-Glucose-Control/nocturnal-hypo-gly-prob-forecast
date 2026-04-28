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
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    # Loss function for fine-tuning.
    # 'pinball'               — pinball/quantile loss on all quantile heads (supervises calibration directly)
    # 'mse'                   — MSE on mean head only (original behaviour)
    # 'joint'                 — pinball on quantile heads + MSE on mean head
    # 'dilate'                — DILATE (shape + temporal distortion) on mean head
    # 'dilate_pinball'        — pinball on all quantile heads + DILATE on every quantile trajectory
    # 'dilate_pinball_median' — pinball on all quantile heads + DILATE on median (0.5) trajectory only
    loss_fn: str = "pinball"
    # DILATE hyper-parameters (ignored unless loss_fn in {'dilate', 'dilate_pinball', 'dilate_pinball_median'})
    dilate_alpha: float = (
        0.5  # shape vs. temporal weight (1=pure shape, 0=pure temporal)
    )
    dilate_gamma: float = 0.01  # soft-DTW smoothing
    # Weight applied to the DILATE term when loss_fn is a pinball+DILATE variant
    # ('dilate_pinball' or 'dilate_pinball_median'); pinball weight = 1.0
    dilate_weight: float = 0.5
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

    # In-training eval callback (writes epoch_metrics.csv to the artifact dir)
    eval_during_training: bool = True
    # Fraction of each patient series reserved (chronologically, from the end)
    # for the in-training eval callback.  Split occurs on the raw series BEFORE
    # windowing to avoid context-window leakage across the boundary.
    eval_temporal_frac: float = 0.10
    # Maximum number of eval windows to use per epoch (None = all).  Set to a
    # few hundred to cap callback runtime without losing metric accuracy.
    eval_subsample: Optional[int] = None

    # Data column names
    target_col: str = "bg_mM"
    interval_mins: int = 5  # CGM sampling interval in minutes; used for gap-length calculations (e.g. imputation_threshold_mins // interval_mins)

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

        if self.interval_mins <= 0:
            raise ValueError(
                f"interval_mins must be positive, got {self.interval_mins}"
            )

        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError(
                f"All quantiles must be between 0 and 1, got {self.quantiles}"
            )

        if self.torch_dtype not in ["bfloat16", "float16", "float32"]:
            raise ValueError(
                f"torch_dtype must be one of ['bfloat16', 'float16', 'float32'], "
                f"got {self.torch_dtype}"
            )

        if self.loss_fn not in [
            "pinball",
            "mse",
            "joint",
            "dilate",
            "dilate_pinball",
            "dilate_pinball_median",
        ]:
            raise ValueError(
                f"loss_fn must be one of ['pinball', 'mse', 'joint', 'dilate', 'dilate_pinball', 'dilate_pinball_median'], got {self.loss_fn}"
            )

        if not 0.0 < self.eval_temporal_frac < 1.0:
            raise ValueError(
                f"eval_temporal_frac must be in (0, 1), got {self.eval_temporal_frac}"
            )

        if self.eval_subsample is not None and self.eval_subsample <= 0:
            raise ValueError(
                f"eval_subsample must be a positive int or None, got {self.eval_subsample}"
            )
