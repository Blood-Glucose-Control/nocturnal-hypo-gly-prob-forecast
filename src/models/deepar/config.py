"""
DeepAR model configuration.

DeepAR is an autoregressive deep learning model based on recurrent neural networks
(LSTM/GRU). It is natively probabilistic — each forecast step predicts the
parameters of a distribution (default: StudentT), yielding well-calibrated
quantile forecasts without residual synthesis.

Memory estimates on Blackwell 6000 (96 GB VRAM):
  batch_size=128, hidden_size=64, num_layers=2 -> ~4 GB peak -> ~24 parallel workers
  For our sweep: 2 workers per GPU is safe, leaving ample headroom.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class DeepARConfig(ModelConfig):
    """Configuration for DeepAR via AutoGluon's TimeSeriesPredictor.

    Architecture defaults are sized conservatively for efficient training on CGM data
    (hundreds of patients, 5-min cadence, 8-hour forecast horizon).
    """

    model_type: str = "deepar"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # RNN architecture (AutoGluon's PyTorch DeepAR is LSTM-only;
    # cell_type and distr_output (as a string) are not exposed by the API).
    hidden_size: int = 64  # LSTM hidden state dimension
    num_layers: int = 2  # Number of stacked LSTM layers
    dropout_rate: float = 0.1

    # Training
    lr: float = 1e-3
    num_batches_per_epoch: int = 50
    batch_size: int = 128
    max_epochs: int = 100
    early_stopping_patience: int = 20
    gradient_clip_val: float = 10.0

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
            "DeepAR": {
                "context_length": self.context_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
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
