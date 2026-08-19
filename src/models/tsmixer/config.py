# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""TSMixer model configuration for Darts-backed training/inference."""

from dataclasses import dataclass, field
from typing import List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class TSMixerConfig(ModelConfig):
    """Configuration for Darts TSMixerModel integration."""

    model_type: str = "tsmixer"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # TSMixer architecture (Darts TSMixerModel)
    hidden_size: int = 64
    ff_size: int = 64
    num_blocks: int = 2
    activation: str = "ReLU"
    dropout: float = 0.1
    norm_type: str = "LayerNorm"
    normalize_before: bool = False
    use_static_covariates: bool = False
    random_state: int = 42

    # Data + training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"
    interval_mins: int = 5
    covariate_cols: List[str] = field(default_factory=list)

    # Gap handling
    imputation_threshold_mins: int = 45
    min_segment_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length
