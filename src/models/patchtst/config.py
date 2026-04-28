# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
PatchTST model configuration.

PatchTST divides the input time series into non-overlapping patches and feeds
them into a Transformer encoder, treating each patch as a token. It is
natively probabilistic when configured with a distributional output head.

Memory estimates on Blackwell 6000 (96 GB VRAM):
  batch_size=256, d_model=128, nhead=16, num_layers=3 -> ~6 GB peak
  2 workers per GPU is safe.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class PatchTSTConfig(ModelConfig):
    """Configuration for PatchTST via AutoGluon's TimeSeriesPredictor."""

    model_type: str = "patchtst"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # Patch Transformer architecture (only fields exposed by AutoGluon's
    # PatchTSTModel API are passed through; ffn_dim_multiplier and dropout
    # are not part of the AG signature and are silently ignored).
    patch_len: int = 16  # Length of each input patch (in time steps)
    stride: int = 8  # Hop between consecutive patches
    d_model: int = 128  # Transformer hidden dimension
    nhead: int = 16  # Number of attention heads (must divide d_model)
    num_encoder_layers: int = 3  # Transformer encoder depth

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-8
    num_batches_per_epoch: int = 100
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 20

    # Minimal segment length to keep after gap handling
    min_segment_length: Optional[int] = None

    # Gap handling
    imputation_threshold_mins: int = 45

    # Covariate config
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
            "PatchTST": {
                "context_length": self.context_length,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "num_batches_per_epoch": self.num_batches_per_epoch,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            }
        }
