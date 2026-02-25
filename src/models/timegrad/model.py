"""TimeGrad model implementation using the base TSFM framework."""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from pts.model.time_grad import TimeGradEstimator
from pts import Trainer
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)

from src.models.timegrad.config import TimeGradConfig
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.utils.logging_helper import info_print

logger = logging.getLogger(__name__)

# GluonTS 0.9.x uses Timestamp.freq which pandas has deprecated. These are
# upstream issues in the pinned gluonts version and cannot be fixed without a
# breaking upgrade, so suppress them here to keep output clean.
warnings.filterwarnings(
    "ignore",
    message="Timestamp.freq is deprecated",
    category=FutureWarning,
    module=r"gluonts\.",
)

# Target dimension is always 1 (univariate blood glucose)
_TARGET_DIM = 1


def _apply_univariate_patch():
    """Monkey-patch pytorchts for univariate (target_dim=1) compatibility.

    TimeGrad's denoising network uses circular-padded Conv1d layers that fail
    when the sequence length (= target_dim) is 1 on PyTorch 2.x, and a
    CondUpsampler whose intermediate dim floors to 0. Patch both at import
    time so no installed files are modified.
    """
    import pts.model.time_grad.epsilon_theta as _et

    _CondUp_init = _et.CondUpsampler.__init__

    def _patched_cond_init(self, cond_length, target_dim):
        nn.Module.__init__(self)
        intermediate = max(target_dim // 2, 1)
        self.linear1 = nn.Linear(cond_length, intermediate)
        self.linear2 = nn.Linear(intermediate, target_dim)

    _et.CondUpsampler.__init__ = _patched_cond_init

    _EpsTheta_init = _et.EpsilonTheta.__init__

    def _patched_eps_init(self, *args, **kwargs):
        _EpsTheta_init(self, *args, **kwargs)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and m.padding_mode == "circular":
                m.padding_mode = "zeros"

    _et.EpsilonTheta.__init__ = _patched_eps_init


def _compute_input_size(freq: str, target_dim: int = 1) -> int:
    """Compute TimeGrad's GRU input_size from the frequency string.

    Formula: target_dim * n_lags + target_dim * embed_dim + 2 * n_fourier_features
    where embed_dim=1 (hardcoded in TimeGradTrainingNetwork) and each
    FourierDateFeature produces 2 dims (sin + cos).
    """
    lags_seq = lags_for_fourier_time_features_from_frequency(freq)
    time_feats = fourier_time_features_from_frequency(freq)
    embed_dim = 1  # Hardcoded in TimeGrad's network
    return target_dim * len(lags_seq) + target_dim * embed_dim + 2 * len(time_feats)


# Apply patch once at module load time
_apply_univariate_patch()


class TimeGradForecaster(BaseTimeSeriesFoundationModel):
    """TimeGrad forecaster: GRU encoder + denoising diffusion head.

    Trains from scratch on blood glucose time series data. Produces
    probabilistic forecasts via diffusion sampling.
    """

    config: TimeGradConfig

    def __init__(
        self, config: TimeGradConfig, lora_config=None, distributed_config=None
    ):
        super().__init__(config, lora_config, distributed_config)

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    def _initialize_model(self) -> None:
        """Build the TimeGradEstimator (predictor is set after training or loading)."""
        info_print("Initializing TimeGrad estimator...")

        use_cuda = torch.cuda.is_available() and not getattr(
            self.config, "use_cpu", False
        )
        self.device = "cuda" if use_cuda else "cpu"

        self.input_size = _compute_input_size(self.config.freq, _TARGET_DIM)
        self.estimator = TimeGradEstimator(
            target_dim=_TARGET_DIM,
            prediction_length=self.config.forecast_length,
            context_length=self.config.context_length,
            cell_type=self.config.cell_type,
            input_size=self.input_size,
            freq=self.config.freq,
            loss_type=self.config.loss_type,
            scaling=self.config.scaling,
            diff_steps=self.config.diff_steps,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            num_layers=self.config.num_layers,
            num_cells=self.config.num_cells,
            residual_layers=self.config.residual_layers,
            residual_channels=self.config.residual_channels,
            num_parallel_samples=self.config.num_samples,
            trainer=Trainer(
                device=self.device,
                epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                num_batches_per_epoch=self.config.num_batches_per_epoch,
                batch_size=self.config.batch_size,
            ),
        )
        self.predictor: Optional[PyTorchPredictor] = None

        info_print(
            f"TimeGrad estimator ready (input_size={self.input_size}, "
            f"device={self.device})"
        )

    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions given context data.

        Args:
            data: DataFrame with 'bg_mM' column containing context window.
            **kwargs: Optional overrides (num_samples).

        Returns:
            Forecast as 1D numpy array of shape (forecast_length,).
        """
        if self.predictor is None:
            raise ValueError(
                "Model has no trained predictor. Call fit() or load() first."
            )

        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")

        context = data[bg_col].values.astype(np.float64)

        # TimeGrad expects 2D targets: (target_dim, timesteps)
        context_2d = context.reshape(1, -1)

        # Build a single-entry ListDataset for inference
        start = pd.Timestamp(data["datetime"].iloc[0])

        test_ds = ListDataset(
            [{"target": context_2d, "start": start}],
            freq=self.config.freq,
            one_dim_target=False,
        )

        num_samples = kwargs.get("num_samples", self.config.num_samples)
        forecast_it = self.predictor.predict(test_ds, num_samples=num_samples)
        forecast = next(forecast_it)

        # samples shape: (num_samples, prediction_length, target_dim)
        samples = forecast.samples.squeeze(-1)  # -> (num_samples, prediction_length)
        return np.median(samples, axis=0)

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """Train TimeGrad on blood glucose data.

        Args:
            train_data: DataFrame with 'bg_mM' and 'p_num' columns,
                        or a pre-built GluonTS ListDataset.
            output_dir: Directory for saving outputs.

        Returns:
            Dict with training metrics.
        """
        info_print("Preparing training data for TimeGrad...")

        if isinstance(train_data, pd.DataFrame):
            dataset = self._dataframe_to_list_dataset(train_data)
        else:
            dataset = train_data

        info_print(
            f"Training TimeGrad for {self.config.num_epochs} epochs "
            f"({self.config.num_batches_per_epoch} batches/epoch)..."
        )

        self.predictor = self.estimator.train(
            dataset,
            num_workers=0,
            prefetch_factor=None,  # type: ignore[arg-type]
        )

        info_print("TimeGrad training complete.")
        return {
            "train_metrics": {
                "status": "complete",
                "epochs": self.config.num_epochs,
                "batches_per_epoch": self.config.num_batches_per_epoch,
            }
        }

    def _dataframe_to_list_dataset(self, df: pd.DataFrame) -> ListDataset:
        """Convert a DataFrame with bg_mM + p_num into a GluonTS ListDataset."""
        bg_col = "bg_mM"
        patient_col = "p_num"

        entries = []
        for pid, group in df.groupby(patient_col):
            bg = group[bg_col].values.astype(np.float64)

            if len(bg) < self.config.context_length + self.config.forecast_length:
                logger.warning(
                    f"Skipping patient {pid}: only {len(bg)} steps "
                    f"(need {self.config.context_length + self.config.forecast_length})"
                )
                continue

            # Determine start timestamp
            start = pd.Timestamp(group["datetime"].iloc[0])

            # TimeGrad multivariate API: (target_dim, timesteps)
            entries.append({"target": bg.reshape(1, -1), "start": start})

        info_print(f"Built ListDataset with {len(entries)} patient time series")
        return ListDataset(entries, freq=self.config.freq, one_dim_target=False)

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Not used — TimeGrad handles data loading internally via GluonTS."""
        raise NotImplementedError(
            "TimeGrad uses GluonTS data loading internally. "
            "Use _train_model() directly."
        )

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save the trained predictor weights to disk."""
        if self.predictor is None:
            return

        checkpoint_dir = Path(output_dir) / "timegrad_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.predictor.prediction_net.state_dict(),
            checkpoint_dir / "state_dict.pt",
        )
        info_print(f"TimeGrad weights saved to {checkpoint_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load trained predictor weights from disk.

        Rebuilds the full predictor from self.estimator (which has all the
        architecture params from config), then loads the saved weights.
        """
        checkpoint_dir = Path(model_dir) / "timegrad_checkpoint"
        weights_path = checkpoint_dir / "state_dict.pt"

        if not weights_path.exists():
            logger.warning(f"No TimeGrad weights found at {weights_path}")
            return

        device = torch.device(self.device)

        # Reconstruct predictor structure from estimator
        transformation = self.estimator.create_transformation()
        dummy_net = self.estimator.create_training_network(device)
        self.predictor = self.estimator.create_predictor(
            transformation, dummy_net, device
        )

        # Load actual trained weights
        self.predictor.prediction_net.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        info_print(f"TimeGrad predictor loaded from {checkpoint_dir}")
