"""
Moment model implementation using the base TSFM framework.

This module provides a concrete implementation of Moment that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments

# Local imports
from src.data.models import ColumnNames
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.moment.config import MomentConfig
from src.utils.logging_helper import info_print, error_print

# MOMENT sequence length limit (from notebook)
MOMENT_MAX_LEN = 512


def _optional_moment_import():
    """Import MOMENTPipeline if momentfm is installed."""
    try:
        from momentfm import MOMENTPipeline
        return MOMENTPipeline
    except ImportError as e:
        raise ImportError(
            "MOMENT model requires the 'momentfm' package. "
            "Install with: pip install momentfm"
        ) from e


class _ContextTargetDataset(Dataset):
    """Dataset of (context, target) arrays for Moment forecasting."""

    def __init__(
        self,
        contexts: List[np.ndarray],
        targets: List[np.ndarray],
        context_lengths: Optional[List[int]] = None,
    ):
        self.contexts = contexts
        self.targets = targets
        self.context_lengths = context_lengths or [c.shape[0] for c in contexts]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "context": self.contexts[idx].astype(np.float32),
            "target": self.targets[idx].astype(np.float32),
            "context_len": int(self.context_lengths[idx]),
        }


class MomentForecaster(BaseTimeSeriesFoundationModel):
    """Moment forecaster implementation using the base TSFM framework.

    Moment is a time series foundation model (CMU) with transformer-based
    masked reconstruction. This implementation supports zero-shot forecasting
    via MOMENTPipeline.forecast() (context + mask -> impute future).

    Attributes:
        config: Moment-specific configuration (MomentConfig instance).
        preprocessor: Unused; scaling is per-context in forecast.
        column_specifiers: Unused; target column from config.
    """

    def __init__(
        self,
        config: MomentConfig,
        lora_config=None,
        distributed_config=None,
    ):
        if not isinstance(config, MomentConfig):
            essential_params = {
                "model_path": getattr(config, "model_path", "AutonLab/MOMENT-1-large"),
                "context_length": getattr(config, "context_length", 512),
                "forecast_length": getattr(config, "forecast_length", 96),
                "learning_rate": getattr(config, "learning_rate", 1e-4),
                "batch_size": getattr(config, "batch_size", 32),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = MomentConfig(**essential_params)

        super().__init__(config, lora_config, distributed_config)
        self.config: MomentConfig = self.config
        self.preprocessor = None
        self.column_specifiers = None
        self._device = torch.device("cuda" if torch.cuda.is_available() and not self.config.use_cpu else "cpu")
        self._target_col: Optional[str] = None

    # --- Properties (base contract) ---
    @property
    def training_backend(self) -> TrainingBackend:
        # Check training_mode for zero-shot, otherwise use training_backend from config
        if hasattr(self.config, "training_mode") and self.config.training_mode == "zero_shot":
            return TrainingBackend.TRANSFORMERS  # Zero-shot uses transformers backend
        return getattr(
            self.config,
            "training_backend",
            TrainingBackend.TRANSFORMERS,
        )

    @property
    def supports_lora(self) -> bool:
        return True

    # --- Model init ---
    def _initialize_model(self) -> None:
        if not self.config.model_path:
            raise ValueError("model_path must be specified in config")

        info_print(f"Initializing Moment model from {self.config.model_path}")

        MOMENTPipeline = _optional_moment_import()

        try:
            self.model = MOMENTPipeline.from_pretrained(
                self.config.model_path,
                model_kwargs={"enable_gradient_checkpointing": True},
            )
            self.model.init()
            self.model.to(self._device)
            self.model.eval()
            # Zero-shot: model is ready for inference without fit()
            self.is_fitted = True
            info_print("Moment model initialized (zero-shot ready)")
        except Exception as e:
            error_print(f"Failed to initialize Moment model: {str(e)}")
            raise

    # --- Forecast helpers (notebook logic) ---
    def _forecast_single(
        self,
        context: np.ndarray,
        prediction_length: int,
        scaler: Optional[Any] = None,
    ) -> np.ndarray:
        """Single univariate forecast: scale context, mask future, impute, unscale."""
        from sklearn.preprocessing import StandardScaler

        scaler = scaler or StandardScaler()
        context_flat = np.asarray(context, dtype=np.float64).reshape(-1, 1)
        context_scaled = scaler.fit_transform(context_flat).flatten()  # type: ignore[union-attr]

        full_len = len(context_scaled) + prediction_length
        if full_len > MOMENT_MAX_LEN:
            keep = MOMENT_MAX_LEN - prediction_length
            context_scaled = context_scaled[-keep:]
            full_len = MOMENT_MAX_LEN

        input_seq = np.zeros(full_len, dtype=np.float32)
        input_seq[: len(context_scaled)] = context_scaled

        # [1, 1, T]
        x_enc = (
            torch.tensor(input_seq, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )
        # [1, T]: 1 = observed, 0 = to predict
        input_mask = torch.ones(full_len, dtype=torch.long, device=self._device)
        input_mask[len(context_scaled) :] = 0
        input_mask = input_mask.unsqueeze(0)

        if self.model is None:
            raise ValueError("Model not initialized")
        with torch.no_grad():
            output = self.model.forecast(x_enc=x_enc, input_mask=input_mask)  # type: ignore[call-overload]

        forecast_scaled = output.forecast.cpu().numpy()[0, 0, :]  # type: ignore[union-attr, call-overload]
        if len(forecast_scaled) > prediction_length:
            forecast_scaled = forecast_scaled[-prediction_length:]
        pred = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()  # type: ignore[union-attr]
        return pred.astype(np.float32)

    def _forecast_batch(
        self,
        contexts: np.ndarray,
        prediction_length: int,
        context_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch of forecasts (loop over batch to match notebook scaling)."""
        from sklearn.preprocessing import StandardScaler

        preds = []
        for i in range(contexts.shape[0]):
            ctx = np.asarray(contexts[i], dtype=np.float64)
            if context_lengths is not None and i < len(context_lengths):
                L = int(context_lengths[i])
                if L < ctx.shape[0]:
                    ctx = ctx[-L:]
            pred = self._forecast_single(ctx, prediction_length, scaler=StandardScaler())
            preds.append(pred)
        return np.stack(preds, axis=0)

    # --- Data: context/target pairs ---
    def _get_target_column(self, df: pd.DataFrame) -> str:
        if self._target_col is not None and self._target_col in df.columns:
            return self._target_col
        col = getattr(
            self.config.data_config,
            "target_features",
            [None],
        )
        if col and len(col) > 0 and col[0] is not None and col[0] in df.columns:
            self._target_col = col[0]
            return col[0]
        bg = ColumnNames.BG.value
        if bg in df.columns:
            self._target_col = bg
            return bg
        raise ValueError(
            f"Target column not found in DataFrame. Columns: {list(df.columns)}"
        )

    def _get_context_target_pairs(
        self,
        data: Any,
        context_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        require_target: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build list of (context, target) from dataset name, DataFrame, or dict of DataFrames."""
        ctx_len = context_length or self.config.context_length
        fcast_len = forecast_length or self.config.forecast_length

        if isinstance(data, str):
            from src.data.diabetes_datasets.data_loader import get_loader
            # Type checker sees overloads with Literal types, but we accept any str at runtime
            loader = get_loader(  # type: ignore[call-overload, assignment]
                data_source_name=data,  # type: ignore[arg-type]
                dataset_type="train",
                use_cached=True,
            )
            # Use validation data for eval-style usage
            if hasattr(loader, "validation_data") and loader.validation_data:
                data = loader.validation_data
            else:
                data = getattr(loader, "processed_data", None) or getattr(loader, "train_data", None)
            if data is None:
                raise ValueError(f"No data available from loader for {data}")

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        if isinstance(data, dict):
            # patient_id -> DataFrame
            from src.data.preprocessing.time_processing import iter_daily_context_forecast_splits
            for _pid, patient_df in data.items():
                target_col = self._get_target_column(patient_df)
                for daytime, nocturnal in iter_daily_context_forecast_splits(patient_df):
                    ctx = daytime[target_col].values
                    tgt = nocturnal[target_col].values[:fcast_len]
                    if np.isnan(ctx).any() or np.isnan(tgt).any():
                        continue
                    if len(ctx) < 10 or len(tgt) < fcast_len:
                        continue
                    if len(ctx) > ctx_len:
                        ctx = ctx[-ctx_len:]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
        elif isinstance(data, pd.DataFrame):
            target_col = self._get_target_column(data)
            if not require_target:
                # Single long series: last ctx_len -> predict next fcast_len
                vals = data[target_col].dropna().values
                if len(vals) >= ctx_len + fcast_len:
                    ctx = vals[-ctx_len - fcast_len : -fcast_len]
                    tgt = vals[-fcast_len:]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
            else:
                from src.data.preprocessing.time_processing import iter_daily_context_forecast_splits

                if ColumnNames.P_NUM.value in data.columns:
                    for _pid, patient_df in data.groupby(ColumnNames.P_NUM.value):
                        for daytime, nocturnal in iter_daily_context_forecast_splits(patient_df):
                            ctx = daytime[target_col].values
                            tgt = nocturnal[target_col].values[:fcast_len]
                            if np.isnan(ctx).any() or np.isnan(tgt).any():
                                continue
                            if len(ctx) < 10 or len(tgt) < fcast_len:
                                continue
                            if len(ctx) > ctx_len:
                                ctx = ctx[-ctx_len:]
                            pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
                else:
                    # Single patient
                    for daytime, nocturnal in iter_daily_context_forecast_splits(data):
                        ctx = daytime[target_col].values
                        tgt = nocturnal[target_col].values[:fcast_len]
                        if np.isnan(ctx).any() or np.isnan(tgt).any():
                            continue
                        if len(ctx) < 10 or len(tgt) < fcast_len:
                            continue
                        if len(ctx) > ctx_len:
                            ctx = ctx[-ctx_len:]
                        pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
        else:
            raise ValueError(
                "data must be a dataset name (str), a DataFrame, or a dict of DataFrames"
            )

        return pairs

    def _prepare_training_data(
        self,
        train_data: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Build a DataLoader of (context, target) batches for prediction/eval."""
        pairs = self._get_context_target_pairs(train_data, require_target=True)
        if not pairs:
            raise ValueError("No (context, target) pairs produced from train_data")

        contexts = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]
        context_lengths = [c.shape[0] for c in contexts]
        max_ctx = max(context_lengths)
        fcast_len = targets[0].shape[0]
        ctx_padded = np.zeros((len(contexts), max_ctx), dtype=np.float32)
        tgt_stacked = np.zeros((len(targets), fcast_len), dtype=np.float32)
        for i, (c, t) in enumerate(zip(contexts, targets)):
            ctx_padded[i, -len(c) :] = c
            tgt_stacked[i] = t

        dataset = _ContextTargetDataset(
            [ctx_padded[i] for i in range(len(ctx_padded))],
            [tgt_stacked[i] for i in range(len(tgt_stacked))],
            context_lengths=context_lengths,
        )
        bs = batch_size if batch_size is not None else self.config.batch_size
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )
        return loader, None, None

    def _extract_ground_truth(self, test_data: Any) -> np.ndarray:
        """Extract ground truth targets in same order as _prepare_training_data."""
        pairs = self._get_context_target_pairs(test_data, require_target=True)
        if not pairs:
            return np.array([])
        return np.concatenate([p[1] for p in pairs], axis=0)

    # --- Public API ---
    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        return_dict: bool = False,
    ) -> np.ndarray:
        """Make zero-shot predictions. Model must be initialized (no fit required).
        
        Note: Base class signature returns np.ndarray only. Use predict_with_metadata()
        for dict return with additional info.
        """
        if self.model is None:
            raise ValueError("Model not initialized; call constructor with valid config")

        bs = batch_size if batch_size is not None else self.config.batch_size
        loader, _, _ = self._prepare_training_data(data, batch_size=bs)
        all_preds = []
        for batch in loader:
            ctx = batch["context"]
            if isinstance(ctx, torch.Tensor):
                ctx = ctx.numpy()
            ctx_lens = batch.get("context_len")
            if isinstance(ctx_lens, torch.Tensor):
                ctx_lens = ctx_lens.numpy()
            preds = self._forecast_batch(
                ctx,
                self.config.forecast_length,
                context_lengths=ctx_lens,
            )
            all_preds.append(preds)
        predictions = np.concatenate(all_preds, axis=0)

        if return_dict:
            # For compatibility, but base signature doesn't support this
            # Users should use predict_with_metadata() if they need dict
            return predictions  # type: ignore[return-value]
        return predictions
    
    def predict_with_metadata(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make predictions and return dict with metadata."""
        predictions = self.predict(data, batch_size=batch_size, return_dict=False)
        return {
            "predictions": predictions,
            "model_type": "moment",
            "config": self.config.to_dict(),
        }

    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Zero-shot predictions without requiring fit(). Same as predict() for Moment."""
        return self.predict(data, batch_size=batch_size)

    # --- Checkpoints ---
    def _save_checkpoint(self, output_dir: str) -> None:
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore[call-overload]
            info_print(f"Moment checkpoint saved to {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        MOMENTPipeline = _optional_moment_import()
        self.model = MOMENTPipeline.from_pretrained(model_dir)
        self.model.init()
        self.model.to(self._device)
        self.model.eval()
        self.is_fitted = True
        info_print(f"Moment checkpoint loaded from {model_dir}")

    # --- Training (no-op for zero-shot) ---
    def _train_model(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Moment is used zero-shot; fine-tuning not implemented."""
        # Check training_mode for zero-shot (TTM pattern)
        if hasattr(self.config, "training_mode") and self.config.training_mode == "zero_shot":
            info_print("Moment zero-shot mode: skipping training")
            return {"train_metrics": {}, "training_history": {}}
        raise NotImplementedError(
            "Moment fine-tuning is not implemented. Use zero-shot: "
            "create_moment_zero_shot_config() and predict_zero_shot()."
        )

    def _get_training_args(self) -> TrainingArguments:
        output_dir = getattr(self.config, "output_dir", None) or "./moment_output"
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",  # Fixed: evaluation_strategy -> eval_strategy
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available() and not self.config.use_cpu,
        )


def create_moment_model(
    model_path: str = "AutonLab/MOMENT-1-large",
    context_length: int = 512,
    forecast_length: int = 96,
    use_lora: bool = False,
    **kwargs: Any,
) -> MomentForecaster:
    """Factory to create a Moment model with sensible defaults."""
    config = MomentConfig(
        model_path=model_path,
        context_length=context_length,
        forecast_length=forecast_length,
        use_lora=use_lora,
        **kwargs,
    )
    return MomentForecaster(config)
