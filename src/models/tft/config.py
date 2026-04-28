# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Temporal Fusion Transformer (TFT) model configuration.

TFT is a sophisticated attention-based model that can incorporate static
metadata, known future covariates, and past observed inputs. In our setting
we use only past observed features (IOB as a past covariate), since future
IOB/COB values are unknowable at the midnight prediction origin.

TFT is natively probabilistic via quantile regression.

Memory estimates on Blackwell 6000 (96 GB VRAM):
  batch_size=256, hidden_size=64, num_heads=4 -> ~8 GB peak
  2 workers per GPU is safe.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class TFTConfig(ModelConfig):
    """Configuration for TemporalFusionTransformer via AutoGluon."""

    model_type: str = "tft"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # TFT architecture
    hidden_size: int = 64  # LSTM encoder/decoder hidden dimension
    num_heads: int = 4  # Attention heads in interpretable multi-head attention
    dropout: float = 0.1
    num_outputs: int = 1  # Target channels (BG only)

    # Training
    lr: float = 1e-3
    num_batches_per_epoch: int = 50
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 20
    gradient_clip_val: float = 1.0

    # Minimal segment length to keep after gap handling
    min_segment_length: Optional[int] = None

    # Gap handling
    imputation_threshold_mins: int = 45

    # Covariate config
    # IOB/COB are past-only features (not known at forecast origin).
    covariate_cols: List[str] = field(default_factory=list)
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"
    interval_mins: int = 5

    # AutoGluon settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    time_limit: Optional[int] = None

    # Quantile levels for probabilistic output
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length

    def get_autogluon_hyperparameters(self) -> Dict:
        """Build hyperparameters dict for TimeSeriesPredictor.fit()."""
        return {
            "TemporalFusionTransformer": {
                "context_length": self.context_length,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout,
                "lr": self.lr,
                "num_batches_per_epoch": self.num_batches_per_epoch,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "trainer_kwargs": {
                    "gradient_clip_val": self.gradient_clip_val,
                },
            }
        }
