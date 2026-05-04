"""
TiDE model configuration.

Extends ModelConfig with TiDE / AutoGluon-specific parameters.
TiDE (Time-series Dense Encoder) is a pure MLP model wrapped in AutoGluon's
TimeSeriesPredictor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class TiDEConfig(ModelConfig):
    """Configuration for TiDE model using AutoGluon's TimeSeriesPredictor.

    Inherits from ModelConfig and adds TiDE-specific attributes for
    AutoGluon training, gap handling, and covariate configuration.

    Critical constraints:
      - encoder_hidden_dim MUST equal decoder_hidden_dim
      - scaling MUST be "mean" (MeanScaler prevents discontinuity)
    """

    # Override parent defaults
    model_type: str = "tide"
    forecast_length: int = 72  # 6 hours at 5-min intervals
    context_length: int = 512  # ~42.7 hours at 5-min intervals
    training_backend: TrainingBackend = TrainingBackend.CUSTOM
    training_mode: str = "from_scratch"

    # TiDE architecture
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    temporal_hidden_dim: int = 256
    num_layers_encoder: int = 2
    num_layers_decoder: int = 2
    distr_hidden_dim: int = 8
    dropout: float = 0.1
    scaling: str = "mean"  # MeanScaler prevents discontinuity

    # Training
    lr: float = 0.000931
    num_batches_per_epoch: int = 300
    batch_size: int = 256
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"

    # Gap handling
    imputation_threshold_mins: int = 45
    min_segment_length: Optional[int] = None

    # Additional TiDE architecture params
    layer_norm: bool = True
    decoder_output_dim: int = 16

    # Training limits
    early_stopping_patience: int = 20
    max_epochs: int = 100

    # Covariates
    covariate_cols: List[str] = field(default_factory=list)
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"
    interval_mins: int = 5

    # AutoGluon settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    time_limit: Optional[int] = None

    def __post_init__(self):
        if self.encoder_hidden_dim != self.decoder_hidden_dim:
            raise ValueError(
                f"TiDE requires encoder_hidden_dim == decoder_hidden_dim, "
                f"got {self.encoder_hidden_dim} != {self.decoder_hidden_dim}. "
                f"This is a hard architectural constraint (see GluonTS source)."
            )
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length

    def get_autogluon_hyperparameters(self) -> Dict:
        """Build hyperparameters dict for TimeSeriesPredictor.fit().

        Returns:
            Dict with "TiDE" key mapping to AutoGluon hyperparameters.
        """
        return {
            "TiDE": {
                "context_length": self.context_length,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "temporal_hidden_dim": self.temporal_hidden_dim,
                "num_layers_encoder": self.num_layers_encoder,
                "num_layers_decoder": self.num_layers_decoder,
                "distr_hidden_dim": self.distr_hidden_dim,
                "layer_norm": self.layer_norm,
                "decoder_output_dim": self.decoder_output_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "num_batches_per_epoch": self.num_batches_per_epoch,
                "batch_size": self.batch_size,
                "scaling": self.scaling,
                "early_stopping_patience": self.early_stopping_patience,
                "max_epochs": self.max_epochs,
                "trainer_kwargs": {
                    "gradient_clip_val": self.gradient_clip_val,
                    "precision": self.precision,
                },
            }
        }


def create_default_tide_config(**overrides) -> TiDEConfig:
    """Create a TiDEConfig with validated defaults for from-scratch training.

    Args:
        **overrides: Configuration parameters to override.

    Returns:
        TiDEConfig instance.
    """
    defaults = {
        "context_length": 512,
        "forecast_length": 72,
    }
    defaults.update(overrides)
    return TiDEConfig(**defaults)
