"""
TTM (TinyTimeMixer) configuration.

Single source of truth for TTM defaults. Construct TTMConfig directly and
override with YAML/CLI dicts via standard merge:

    config = TTMConfig(**{**yaml_overrides, **cli_overrides})
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tsfm_public.toolkit.time_series_preprocessor import ScalerType

from ..base import ModelConfig, TrainingBackend

VALID_TTM_TRAINING_MODES = {"zero_shot", "fine_tune", "from_scratch"}
SCALER_TYPE_MAP = {
    "standard": ScalerType.STANDARD,
    "minmax": ScalerType.MINMAX,
}


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
    model_path: Optional[str] = "ibm-granite/granite-timeseries-ttm-r2"
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

    def __post_init__(self) -> None:
        self.validate()

    def get_scaler_type(self) -> ScalerType:
        """Convert string scaler_type to TSFM ScalerType enum."""
        try:
            return SCALER_TYPE_MAP[self.scaler_type]
        except KeyError as exc:
            allowed = ", ".join(sorted(SCALER_TYPE_MAP))
            raise ValueError(
                f"scaler_type must be one of [{allowed}], got {self.scaler_type}"
            ) from exc

    def validate(self) -> None:
        """Validate configuration. Raises ValueError if invalid."""
        errors = []

        if not self.model_path:
            errors.append("model_path is required")
        if self.context_length <= 0:
            errors.append("context_length must be positive")
        if self.forecast_length <= 0:
            errors.append("forecast_length must be positive")
        if self.training_mode not in VALID_TTM_TRAINING_MODES:
            allowed_modes = ", ".join(sorted(VALID_TTM_TRAINING_MODES))
            errors.append(
                f"training_mode must be one of [{allowed_modes}], got {self.training_mode}"
            )
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.num_epochs < 0:
            errors.append("num_epochs must be >= 0")
        if self.resolution_min <= 0:
            errors.append("resolution_min must be positive")
        if self.num_input_channels <= 0:
            errors.append("num_input_channels must be positive")
        if self.num_output_channels <= 0:
            errors.append("num_output_channels must be positive")
        if not (0 <= self.fewshot_percent <= 100):
            errors.append("fewshot_percent must be between 0 and 100")
        if self.input_features is None:
            errors.append("input_features must be a list (can be empty for univariate)")
        if not self.target_features:
            errors.append("target_features cannot be empty")
        if self.split_config:
            required_keys = {"train", "val", "test"}
            missing = required_keys - set(self.split_config)
            extra = set(self.split_config) - required_keys
            if missing:
                errors.append(f"split_config missing required keys: {sorted(missing)}")
            if extra:
                errors.append(f"split_config has unsupported keys: {sorted(extra)}")
            for split_name, split_value in self.split_config.items():
                if split_value < 0 or split_value > 1:
                    errors.append(
                        f"split_config[{split_name}] must be in [0, 1], got {split_value}"
                    )
            split_sum = sum(self.split_config.values())
            if abs(split_sum - 1.0) > 1e-6:
                errors.append(f"split_config must sum to 1.0, got {split_sum}")
        if self.scaler_type not in SCALER_TYPE_MAP:
            allowed_scalers = ", ".join(sorted(SCALER_TYPE_MAP))
            errors.append(
                f"scaler_type must be one of [{allowed_scalers}], got {self.scaler_type}"
            )
        if self.training_mode == "zero_shot" and self.num_epochs != 0:
            errors.append("training_mode=zero_shot requires num_epochs=0")

        if errors:
            raise ValueError(
                "TTMConfig validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )
