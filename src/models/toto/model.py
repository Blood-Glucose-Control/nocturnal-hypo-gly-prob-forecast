"""
Toto model implementation using the base TSFM framework.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from toto.model.toto import Toto
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster as _TotoForecaster

from src.models.toto.config import TotoConfig
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

logger = logging.getLogger(__name__)

INTERVAL_MINS = 5


@ModelRegistry.register("toto")
class TotoForecaster(BaseTimeSeriesFoundationModel):
    """Toto forecaster implementation."""

    config_class = TotoConfig
    config: TotoConfig

    DEFAULT_NUM_SAMPLES: int = 40

    def _initialize_model(self) -> None:
        """Load the Toto model from HuggingFace."""
        info_print("Initializing Toto model from Datadog/Toto-Open-Base-1.0...")

        toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")

        use_cuda = torch.cuda.is_available() and not getattr(
            self.config, "use_cpu", False
        )
        self.device = "cuda" if use_cuda else "cpu"
        toto.to(self.device)

        self.model = toto.model
        self.forecaster = _TotoForecaster(self.model)
        self._patch_size = getattr(self.model.patch_embed, "patch_size", 16)

        info_print(f"Toto model initialized on {self.device}")

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return True

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Shared inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_timestamps(data: pd.DataFrame) -> pd.DatetimeIndex:
        """Get timestamps from DatetimeIndex or 'datetime' column."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index
        if "datetime" in data.columns:
            return pd.DatetimeIndex(pd.to_datetime(data["datetime"]))
        raise ValueError("Data must have a DatetimeIndex or 'datetime' column")

    @staticmethod
    def _timestamps_to_seconds(timestamps: pd.DatetimeIndex) -> torch.Tensor:
        """Convert pandas timestamps to seconds-since-epoch float tensor."""
        return torch.tensor(
            np.asarray(timestamps.astype(np.int64)) // 1_000_000_000,
            dtype=torch.float32,
        )

    def _build_variates(self, data: pd.DataFrame) -> List[torch.Tensor]:
        """Extract BG target + covariate tensors from a single episode."""
        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")
        variates = [torch.tensor(data[bg_col].values, dtype=torch.float32)]
        for col in self.config.covariate_cols or []:
            if col not in data.columns:
                raise ValueError(f"Covariate column '{col}' not found in data")
            variates.append(torch.tensor(data[col].values, dtype=torch.float32))
        return variates

    def _extract_bg_forecast(self, forecast) -> np.ndarray:
        """Extract BG (variate 0) from a Toto Forecast object.

        Returns shape (batch, forecast_length). Uses median when sampling,
        mean otherwise.
        """
        if self.config.num_samples is None:
            return forecast.mean[:, 0, :].cpu().numpy()
        return forecast.median[:, 0, :].cpu().numpy()

    def _run_forecast(self, inputs: MaskedTimeseries) -> np.ndarray:
        """Run forecaster and return BG predictions as numpy array."""
        with torch.no_grad():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=self.config.forecast_length,
                num_samples=self.config.num_samples,
                samples_per_batch=self.config.samples_per_batch,
            )
        return self._extract_bg_forecast(forecast)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict a single episode.

        Returns:
            np.ndarray: A 1D array of shape (forecast_length,) when
            ``quantile_levels`` is ``None``; otherwise a 2D array of shape
            (n_quantiles, forecast_length) containing the requested quantiles.
        """
        timestamps = self._extract_timestamps(data)
        variates = self._build_variates(data)
        num_covariates = len(self.config.covariate_cols or [])

        # (1, num_variates, series_len)
        series = torch.stack(variates, dim=0).unsqueeze(0).to(self.device)
        num_variates = series.shape[1]

        # Broadcast timestamps to (1, num_variates, series_len)
        ts_seconds = (
            self._timestamps_to_seconds(timestamps)
            .unsqueeze(0)
            .expand(num_variates, -1)
            .unsqueeze(0)
            .to(self.device)
        )

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=torch.zeros_like(series),
            timestamp_seconds=ts_seconds,
            time_interval_seconds=torch.full(
                (num_variates,),
                INTERVAL_MINS * 60,
                dtype=torch.float32,
                device=self.device,
            ),
            num_exogenous_variables=num_covariates,
        )

        if quantile_levels is not None:
            num_samples = self.config.num_samples or self.DEFAULT_NUM_SAMPLES
            with torch.no_grad():
                forecast = self.forecaster.forecast(
                    inputs,
                    prediction_length=self.config.forecast_length,
                    num_samples=num_samples,
                    samples_per_batch=self.config.samples_per_batch,
                )
            # samples: (1, num_variates, fh, num_samples) → BG variate → (fh, num_samples)
            samples = forecast.samples[0, 0, :, :].cpu().numpy()
            return np.quantile(samples, quantile_levels, axis=1)  # (n_q, fh)

        return self._run_forecast(inputs).flatten()

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Batched prediction: stack episodes into one forward pass.

        Episodes are left-padded to the longest context length so the most
        recent timesteps are right-aligned. The padding_mask tells the model
        which timesteps are real data.
        """
        covariate_cols = self.config.covariate_cols or []
        num_covariates = len(covariate_cols)
        num_variates = 1 + num_covariates

        # Group episodes and build per-episode tensors
        episode_ids: List[str] = []
        all_series: List[torch.Tensor] = []  # each (num_variates, T)
        all_ts: List[torch.Tensor] = []  # each (T,)

        for ep_id, ep_data in data.groupby(episode_col):
            episode_ids.append(str(ep_id))
            all_ts.append(
                self._timestamps_to_seconds(self._extract_timestamps(ep_data))
            )
            all_series.append(torch.stack(self._build_variates(ep_data), dim=0))

        if not episode_ids:
            return {}

        # Left-pad to max length so recent context is right-aligned
        max_len = max(s.shape[1] for s in all_series)
        batch_size = len(all_series)
        _eval_bs = self.config.eval_batch_size
        if _eval_bs is None:
            chunk = batch_size
        else:
            chunk = int(
                _eval_bs
            )  # raises TypeError or ValueError if unconvertible; accepts numeric strings
            if chunk <= 0:
                raise ValueError(
                    f"eval_batch_size must be a positive integer, got {_eval_bs!r}"
                )

        series = torch.zeros(batch_size, num_variates, max_len)
        padding_mask = torch.zeros(batch_size, num_variates, max_len, dtype=torch.bool)
        ts_batch = torch.zeros(batch_size, num_variates, max_len)

        for i, (s, ts) in enumerate(zip(all_series, all_ts)):
            series_len = s.shape[1]
            pad_start = max_len - series_len
            series[i, :, pad_start:] = s
            padding_mask[i, :, pad_start:] = True
            ts_batch[i, :, pad_start:] = ts.unsqueeze(0).expand(num_variates, -1)

        series = series.to(self.device)
        padding_mask = padding_mask.to(self.device)
        ts_batch = ts_batch.to(self.device)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=torch.zeros_like(series),
            timestamp_seconds=ts_batch,
            time_interval_seconds=torch.full(
                (batch_size, num_variates),
                INTERVAL_MINS * 60,
                dtype=torch.float32,
                device=self.device,
            ),
            num_exogenous_variables=num_covariates,
        )

        if quantile_levels is not None:
            num_samples = self.config.num_samples or self.DEFAULT_NUM_SAMPLES
            results: Dict[str, np.ndarray] = {}
            for start in range(0, batch_size, chunk):
                end = min(start + chunk, batch_size)
                chunk_inputs = MaskedTimeseries(
                    series=inputs.series[start:end],
                    padding_mask=inputs.padding_mask[start:end],
                    id_mask=inputs.id_mask[start:end],
                    timestamp_seconds=inputs.timestamp_seconds[start:end],
                    time_interval_seconds=inputs.time_interval_seconds[start:end],
                    num_exogenous_variables=inputs.num_exogenous_variables,
                )
                with torch.no_grad():
                    forecast = self.forecaster.forecast(
                        chunk_inputs,
                        prediction_length=self.config.forecast_length,
                        num_samples=num_samples,
                        samples_per_batch=self.config.samples_per_batch,
                    )
                # samples: (chunk, num_variates, fh, num_samples) → BG variate → (fh, num_samples)
                for local_i, eid in enumerate(episode_ids[start:end]):
                    results[eid] = np.quantile(
                        forecast.samples[local_i, 0, :, :].cpu().numpy(),
                        quantile_levels,
                        axis=1,
                    )  # (n_q, fh)
            return results

        results_pt: Dict[str, np.ndarray] = {}
        for start in range(0, batch_size, chunk):
            end = min(start + chunk, batch_size)
            chunk_inputs = MaskedTimeseries(
                series=inputs.series[start:end],
                padding_mask=inputs.padding_mask[start:end],
                id_mask=inputs.id_mask[start:end],
                timestamp_seconds=inputs.timestamp_seconds[start:end],
                time_interval_seconds=inputs.time_interval_seconds[start:end],
                num_exogenous_variables=inputs.num_exogenous_variables,
            )
            chunk_preds = self._run_forecast(chunk_inputs)
            for local_i, eid in enumerate(episode_ids[start:end]):
                results_pt[eid] = chunk_preds[local_i]
        return results_pt

    def _dataframe_to_hf_dataset(self, train_data: pd.DataFrame):
        """Convert a flat DataFrame to HuggingFace Dataset format for Toto.

        Each patient's contiguous BG series becomes one row in the HF dataset
        with 'timestamp', 'target', and optional covariate fields.

        Args:
            train_data: DataFrame with 'bg_mM', 'datetime', and 'p_num' columns.

        Returns:
            HuggingFace Dataset with timestamp/target fields per series.
        """
        import datasets as hfds

        patient_col = "p_num" if "p_num" in train_data.columns else "id"
        bg_col = "bg_mM"
        time_col = "datetime"
        covariate_cols = self.config.covariate_cols or []

        records = []
        for pid, group in train_data.groupby(patient_col):
            group = group.sort_values(time_col)
            target = group[bg_col].values.astype(np.float32)

            # Skip series that are too short
            min_len = 3 * self._patch_size + self.config.forecast_length
            if len(target) < min_len:
                logger.warning(
                    "Patient %s has only %d steps (need %d), skipping",
                    pid,
                    len(target),
                    min_len,
                )
                continue

            ts_strings = (
                pd.to_datetime(group[time_col])
                .dt.strftime("%Y-%m-%dT%H:%M:%S")
                .tolist()
            )
            record = {
                "timestamp": ts_strings,
                "target": target.tolist(),
                "freq": f"{INTERVAL_MINS}min",
            }

            # Add covariates as separate fields (each becomes an ev_field)
            if covariate_cols:
                for col in covariate_cols:
                    values = group[col].values.astype(np.float32)
                    # Fill NaNs with 0 for covariates
                    values = np.nan_to_num(values, nan=0.0)
                    record[col] = values.tolist()
            else:
                # Dummy exogenous field required by transform_fev_dataset
                record["feat_dynamic_real"] = np.zeros_like(target).tolist()

            records.append(record)

        if not records:
            raise ValueError("No patient series long enough for fine-tuning")

        info_print(f"Prepared {len(records)} patient series for fine-tuning")
        if covariate_cols:
            info_print(f"  Covariates: {covariate_cols}")
        return hfds.Dataset.from_list(records)

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Prepare data for Toto fine-tuning.

        Converts the flat DataFrame to a HuggingFace Dataset, then wraps it
        in a FinetuneDataModule. Returns (datamodule, None, None) — Lightning
        handles train/val splitting internally.
        """
        from toto.data.datamodule.finetune_datamodule import FinetuneDataModule

        hf_dataset = self._dataframe_to_hf_dataset(train_data)

        # Compute context length aligned to patch size
        max_context_length = self.config.context_length
        if max_context_length is None:
            max_context_length = 8 * self._patch_size

        covariate_cols = self.config.covariate_cols or []
        if covariate_cols:
            ev_fields = list(covariate_cols)
            ev_transform_fns = [lambda x: np.asarray(x, dtype=np.float32)] * len(
                covariate_cols
            )
            add_exogenous = True
        else:
            ev_fields = ["feat_dynamic_real"]
            ev_transform_fns = [lambda x: np.asarray(x, dtype=np.float32)]
            add_exogenous = False

        dm = FinetuneDataModule(
            dataset=hf_dataset,
            max_context_length=max_context_length,
            prediction_horizon=self.config.forecast_length,
            patch_size=self._patch_size,
            train_batch_size=self.config.train_batch_size,
            val_batch_size=self.config.val_batch_size,
            num_workers=0,
            num_train_samples=1,
            add_exogenous_features=add_exogenous,
            target_fields=["target"],
            target_transform_fns=[lambda x: np.asarray(x, dtype=np.float32)],
            ev_fields=ev_fields,
            ev_transform_fns=ev_transform_fns,
        )

        return (dm, None, None)

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune Toto using PyTorch Lightning.

        The base class fit() passes raw train_data here for CUSTOM backends.
        We call _prepare_training_data ourselves.
        """
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
        from lightning.pytorch.loggers import TensorBoardLogger
        from toto.model.lightning_module import TotoForFinetuning

        seed_everything(42, workers=True)

        # Prepare data
        dm, _, _ = self._prepare_training_data(train_data)

        # Create Lightning module from current backbone
        has_covariates = bool(self.config.covariate_cols)
        lightning_module = TotoForFinetuning(
            pretrained_backbone=self.model,
            val_prediction_len=self.config.val_prediction_len,
            stable_steps=self.config.stable_steps,
            decay_steps=self.config.decay_steps,
            warmup_steps=self.config.warmup_steps,
            lr=self.config.lr,
            min_lr=self.config.min_lr,
            add_exogenous_features=has_covariates,
        )
        lightning_module.to(self.device)

        # Checkpointing
        os.makedirs(output_dir, exist_ok=True)
        ckpt_dir = os.path.join(output_dir, "checkpoints")

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="toto_finetuning",
        )

        # Support both max_steps (Toto-native) and num_epochs (generic workflow).
        # If num_epochs is set, use max_epochs; otherwise use max_steps.
        trainer_kwargs = dict(
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
            logger=tb_logger,
        )
        if self.config.num_epochs is not None:
            trainer_kwargs["max_epochs"] = self.config.num_epochs
            info_print(
                f"Starting Toto fine-tuning for {self.config.num_epochs} epoch(s)..."
            )
        else:
            trainer_kwargs["max_steps"] = self.config.max_steps
            info_print(
                f"Starting Toto fine-tuning for {self.config.max_steps} steps..."
            )

        trainer = Trainer(**trainer_kwargs)
        trainer.fit(lightning_module, datamodule=dm)

        best_ckpt = checkpoint_callback.best_model_path
        best_score = (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        )

        # Load best checkpoint weights (not the final overfitted weights)
        if best_ckpt and os.path.exists(best_ckpt):
            info_print(f"Loading best checkpoint: {best_ckpt}")
            best_lightning = TotoForFinetuning.load_from_checkpoint(
                checkpoint_path=best_ckpt,
                pretrained_backbone=lightning_module.model,
                map_location=self.device,
            )
            self.model = best_lightning.model.to(self.device)
        else:
            info_print("No best checkpoint found, using final model weights")
            self.model = lightning_module.model.to(self.device)

        self.forecaster = _TotoForecaster(self.model)

        info_print(f"Fine-tuning complete. Best checkpoint: {best_ckpt}")
        if best_score is not None:
            info_print(f"Best val_loss: {best_score:.4f}")

        return {
            "best_checkpoint": best_ckpt,
            "best_val_loss": best_score,
            "max_steps": self.config.max_steps,
        }

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save the fine-tuned model checkpoint.

        Saves the backbone state dict and a reference JSON for loading.
        """
        if self.model is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, "toto_backbone.pt")
        torch.save(self.model.state_dict(), weights_path)

        ref_path = os.path.join(output_dir, "toto_checkpoint.json")
        with open(ref_path, "w") as f:
            json.dump({"weights_file": "toto_backbone.pt"}, f, indent=2)

        logger.info("Toto checkpoint saved to %s", output_dir)

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load a fine-tuned Toto checkpoint.

        Loads the backbone state dict saved by _save_checkpoint.
        """
        ref_path = os.path.join(model_dir, "toto_checkpoint.json")
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                ref = json.load(f)
            weights_path = os.path.join(model_dir, ref["weights_file"])
        else:
            # Fall back to looking for the weights file directly
            weights_path = os.path.join(model_dir, "toto_backbone.pt")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Toto checkpoint not found at {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.device)
        # Enable variate labels if the checkpoint has them (covariate-trained model)
        if "target_variate_label" in state_dict:
            self.model.enable_variate_labels()
        self.model.load_state_dict(state_dict)
        self.forecaster = _TotoForecaster(self.model)
        self.is_fitted = True

        logger.info("Toto checkpoint loaded from %s", weights_path)
