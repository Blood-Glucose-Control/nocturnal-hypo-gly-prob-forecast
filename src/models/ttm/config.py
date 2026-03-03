"""
TTM (TinyTimeMixer) configuration.

Single source of truth for TTM defaults. No factory functions — construct
TTMConfig directly, override with YAML/CLI dicts via standard merge:

    config = TTMConfig(**{**yaml_overrides, **cli_overrides})
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend
from tsfm_public.toolkit.time_series_preprocessor import ScalerType


@dataclass
class TTMConfig(ModelConfig):
    """TTM model configuration.

    All defaults are declared here. To override, pass keyword arguments:

        config = TTMConfig(num_epochs=5, learning_rate=1e-3)

    For YAML + CLI override pattern:

        overrides = {**yaml_dict, **cli_dict}
        config = TTMConfig(**overrides)

    Zero-shot is just a config with specific values, not a separate class:

        zs = TTMConfig(**{**overrides,
            "training_mode": "zero_shot",
            "freeze_backbone": True,
            "num_epochs": 0,
        })
    """

    # --- TTM identity (override ModelConfig defaults) ---
    model_type: str = "ttm"
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2"
    training_backend: TrainingBackend = TrainingBackend.TRANSFORMERS

    # --- Architecture ---
    context_length: int = 512
    forecast_length: int = 96
    num_input_channels: int = 1  # auto-set from input_features in model.fit()
    num_output_channels: int = 1
    prediction_filter_length: Optional[int] = None

    # --- Training ---
    training_mode: str = "fine_tune"
    freeze_backbone: bool = False
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # --- TTM-specific training ---
    scaler_type: str = "standard"
    loss_function: str = "mse"
    use_tracking_callback: bool = True
    find_optimal_lr: bool = False
    fewshot_percent: int = 5
    logging_dir: Optional[str] = None

    # --- Data features ---
    input_features: List[str] = field(
        default_factory=lambda: [
            "cob",
            "carb_availability",
            "insulin_availability",
            "iob",
            "steps",
        ]
    )
    target_features: List[str] = field(default_factory=lambda: ["bg_mM"])
    resolution_min: int = 5

    # --- Data splitting ---
    split_config: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.9, "val": 0.05, "test": 0.05}
    )

    def get_scaler_type(self) -> ScalerType:
        """Convert string scaler_type to TSFM ScalerType enum."""
        mapping = {
            "standard": ScalerType.STANDARD,
            "minmax": ScalerType.MINMAX,
        }
        return mapping.get(self.scaler_type, ScalerType.STANDARD)

    def validate(self) -> None:
        """Validate configuration. Raises ValueError if invalid."""
        errors = []

        if not self.model_path:
            errors.append("model_path is required")
        if self.context_length <= 0:
            errors.append("context_length must be positive")
        if self.forecast_length <= 0:
            errors.append("forecast_length must be positive")
        if self.input_features is None:
            errors.append("input_features must be a list (can be empty for univariate)")
        if not self.target_features:
            errors.append("target_features cannot be empty")
        if self.split_config:
            split_sum = sum(self.split_config.values())
            if abs(split_sum - 1.0) > 1e-6:
                errors.append(f"split_config must sum to 1.0, got {split_sum}")
        if self.scaler_type not in ("standard", "minmax", "robust"):
            errors.append(
                f"scaler_type must be standard/minmax/robust, got {self.scaler_type}"
            )

        if errors:
            raise ValueError(
                "TTMConfig validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )


# ---------------------------------------------------------------------------
# Backwards compatibility — old names still importable but just call TTMConfig
# ---------------------------------------------------------------------------


def create_default_ttm_config(**overrides) -> TTMConfig:
    """DEPRECATED: Use TTMConfig(**overrides) directly."""
    config = TTMConfig(**overrides)
    config.validate()
    return config


def create_ttm_fine_tuning_config(**overrides) -> TTMConfig:
    """DEPRECATED: Use TTMConfig(training_mode="fine_tune", ...) directly."""
    defaults = {"training_mode": "fine_tune", "learning_rate": 1e-5, "num_epochs": 5}
    return create_default_ttm_config(**{**defaults, **overrides})


def create_ttm_zero_shot_config(**overrides) -> TTMConfig:
    """DEPRECATED: Use TTMConfig(training_mode="zero_shot", freeze_backbone=True, num_epochs=0) directly."""
    forced = {"training_mode": "zero_shot", "freeze_backbone": True, "num_epochs": 0}
    return create_default_ttm_config(**{**overrides, **forced})
