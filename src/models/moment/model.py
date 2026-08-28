"""
Moment model implementation using the base TSFM framework.

This module provides a concrete implementation of Moment that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from ...data.models import ColumnNames
from ...utils.logging_helper import error_print, info_print
from ..base import BaseTimeSeriesFoundationModel, TrainingBackend
from ..base.registry import ModelRegistry
from .config import MomentConfig

# MOMENT sequence length limit (from notebook)
MOMENT_MAX_LEN = 512

# Minimum scale for optional wrapper-side per-window normalization.
# Applied only when config.use_wrapper_normalization=True.
NORMALIZATION_SCALE_FLOOR = 0.1


def _optional_moment_import():
    """Import MOMENTPipeline if momentfm is installed."""
    try:
        from momentfm import MOMENTPipeline  # pyright: ignore[reportMissingImports]

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
            "context": self.contexts[idx],
            "target": self.targets[idx],
            "context_len": int(self.context_lengths[idx]),
        }


@ModelRegistry.register("moment")
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

    config_class = MomentConfig

    def __init__(
        self,
        config: MomentConfig,
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

        # Set device BEFORE super().__init__() because _initialize_model() is called during init
        use_cpu = getattr(config, "use_cpu", False)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        )

        super().__init__(config)
        self.config: MomentConfig = self.config
        self.preprocessor = None
        self.column_specifiers = None
        self._target_col: Optional[str] = None

    # --- Properties (base contract) ---
    @property
    def training_backend(self) -> TrainingBackend:
        # Check training_mode for zero-shot, otherwise use training_backend from config
        if (
            hasattr(self.config, "training_mode")
            and self.config.training_mode == "zero_shot"
        ):
            return TrainingBackend.TRANSFORMERS  # Zero-shot uses transformers backend
        return getattr(
            self.config,
            "training_backend",
            TrainingBackend.PYTORCH,  # Fine-tuning always uses custom PyTorch loop
        )

    @property
    def supports_zero_shot(self) -> bool:
        return True

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return False

    @property
    def _use_wrapper_normalization(self) -> bool:
        return bool(getattr(self.config, "use_wrapper_normalization", False))

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Point forecast implementation delegated to by the base-class predict().

        Moment currently provides deterministic point forecasts only.
        """
        bs = kwargs.get("batch_size") or self.config.batch_size
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
        # Base contract expects shape (forecast_length,) for a single prediction.
        if predictions.ndim == 2 and predictions.shape[0] == 1:
            return predictions[0]
        return predictions

    # --- Model init ---
    def _initialize_model(self) -> None:
        if not self.config.model_path:
            raise ValueError("model_path must be specified in config")

        info_print(f"Initializing Moment model from {self.config.model_path}")

        MOMENTPipeline = _optional_moment_import()

        try:
            model = MOMENTPipeline.from_pretrained(
                self.config.model_path,
                model_kwargs={"enable_gradient_checkpointing": True},
            )
            model = cast(Any, model)
            model.init()
            model.to(self._device)
            model.eval()
            self.model = model
            # Zero-shot: model is ready for inference without fit()
            self.is_fitted = True
            info_print("Moment model initialized (zero-shot ready)")
        except Exception as e:
            error_print(f"Failed to initialize Moment model: {str(e)}")
            raise

    def _forecast_batch(
        self,
        contexts: np.ndarray,
        prediction_length: int,
        context_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch forecasts in a single MOMENT forward pass when possible.

        Contexts are packed into a shared tensor with per-sample masks so
        patient/episode boundaries remain independent within the batch. Optional
        wrapper-side normalization can be enabled via config.
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        contexts = np.asarray(contexts, dtype=np.float32)
        if contexts.ndim == 1:
            contexts = contexts[None, :, None]
        elif contexts.ndim == 2:
            contexts = contexts[:, :, None]
        if contexts.ndim != 3:
            raise ValueError(
                f"contexts must be 1D/2D/3D (batch,time[,channels]), got shape {contexts.shape}"
            )

        batch_size, max_ctx, n_channels = contexts.shape
        if batch_size == 0:
            return np.zeros((0, prediction_length), dtype=np.float32)

        max_allowed_ctx = MOMENT_MAX_LEN - prediction_length
        if max_allowed_ctx <= 0:
            raise ValueError(
                f"prediction_length={prediction_length} must be < {MOMENT_MAX_LEN}"
            )

        if context_lengths is None:
            context_lengths_arr = np.full(batch_size, max_ctx, dtype=np.int64)
        else:
            context_lengths_arr = np.asarray(context_lengths, dtype=np.int64).copy()
            if context_lengths_arr.shape[0] != batch_size:
                raise ValueError(
                    "context_lengths must have same length as contexts batch size"
                )
            context_lengths_arr = np.clip(context_lengths_arr, 1, max_ctx)

        effective_ctx_lens = np.minimum(context_lengths_arr, max_allowed_ctx)
        max_effective_ctx = int(np.max(effective_ctx_lens))
        full_len = max_effective_ctx + prediction_length

        input_seq = np.zeros((batch_size, n_channels, full_len), dtype=np.float32)
        input_mask = np.zeros((batch_size, full_len), dtype=np.int64)
        locs = np.zeros(batch_size, dtype=np.float32)
        scales = np.ones(batch_size, dtype=np.float32)

        for i in range(batch_size):
            raw_len = int(context_lengths_arr[i])
            keep_len = int(effective_ctx_lens[i])
            ctx = contexts[i, -raw_len:, :].astype(np.float64, copy=False)
            if keep_len < raw_len:
                ctx = ctx[-keep_len:, :]

            if self._use_wrapper_normalization:
                loc_vec = np.nanmean(ctx, axis=0)
                scale_vec = np.nanstd(ctx, axis=0)
                loc_vec = np.where(np.isnan(loc_vec), 0.0, loc_vec)
                scale_vec = np.where(
                    np.isnan(scale_vec) | (scale_vec < NORMALIZATION_SCALE_FLOOR),
                    NORMALIZATION_SCALE_FLOOR,
                    scale_vec,
                )
                ctx_model = ((ctx - loc_vec[None, :]) / scale_vec[None, :]).astype(
                    np.float32
                )
                ctx_model = np.nan_to_num(ctx_model, nan=0.0, posinf=0.0, neginf=0.0)
                loc = float(loc_vec[0])
                scale = float(scale_vec[0])
            else:
                loc = 0.0
                scale = 1.0
                fill_vec = np.nanmean(ctx, axis=0)
                fill_vec = np.where(np.isnan(fill_vec), 0.0, fill_vec)
                finite = np.isfinite(ctx)
                ctx_model = np.where(finite, ctx, fill_vec[None, :]).astype(np.float32)

            seq_len = min(ctx_model.shape[0], max_effective_ctx)
            if seq_len <= 0:
                continue

            input_seq[i, :, :seq_len] = ctx_model[-seq_len:, :].T
            input_mask[i, :seq_len] = 1
            locs[i] = np.float32(loc)
            scales[i] = np.float32(scale)

        x_enc = torch.tensor(input_seq, dtype=torch.float32, device=self._device)
        mask = torch.tensor(input_mask, dtype=torch.long, device=self._device)

        with torch.no_grad():
            output = self.model.forecast(x_enc=x_enc, input_mask=mask)  # type: ignore[call-overload]

        forecast_out = output.forecast[:, 0, :].detach().cpu().numpy()  # type: ignore[union-attr]
        if forecast_out.shape[1] > prediction_length:
            forecast_out = forecast_out[:, -prediction_length:]
        elif forecast_out.shape[1] < prediction_length:
            raise ValueError(
                f"Model returned {forecast_out.shape[1]} forecast steps; expected at least {prediction_length}."
            )

        preds = forecast_out * scales[:, None] + locs[:, None]
        return preds.astype(np.float32)

    # --- Data: context/target pairs ---
    def _get_target_column(self, df: pd.DataFrame) -> str:
        if self._target_col is not None and self._target_col in df.columns:
            return self._target_col
        # Use target_col from config if set (e.g. from sweep YAML)
        config_target_col = getattr(self.config, "target_col", None)
        if config_target_col and config_target_col in df.columns:
            self._target_col = config_target_col
            return config_target_col
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

    def _get_covariate_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Return configured covariate columns that are available in df."""
        configured = list(getattr(self.config, "covariate_cols", []) or [])
        covariates = [c for c in configured if c != target_col and c in df.columns]
        return covariates

    def _get_input_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Input channel order for MOMENT: target first, then covariates."""
        return [target_col] + self._get_covariate_columns(df, target_col)

    def _build_context_matrix(
        self, df: pd.DataFrame, input_cols: List[str]
    ) -> np.ndarray:
        """Build finite context matrix [time, channels] from selected columns."""
        mat_df = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        mat_df = mat_df.dropna()
        if mat_df.empty:
            return np.empty((0, len(input_cols)), dtype=np.float32)
        return mat_df.values.astype(np.float32)

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
        max_train_windows = int(getattr(self.config, "max_train_windows", 2048))
        window_stride = int(getattr(self.config, "window_stride", fcast_len))

        def _append_rolling_pairs(
            df: pd.DataFrame,
            target_col: str,
            input_cols: List[str],
        ) -> None:
            """Fallback window builder when midnight splits are unavailable."""
            mat = self._build_context_matrix(df, input_cols)
            if mat.shape[0] < ctx_len + fcast_len:
                return
            target_vals = mat[:, 0]
            start = 0
            stop = mat.shape[0] - (ctx_len + fcast_len) + 1
            stride = max(1, window_stride)
            for i in range(start, stop, stride):
                ctx = mat[i : i + ctx_len, :]
                tgt = target_vals[i + ctx_len : i + ctx_len + fcast_len]
                pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
                if require_target and len(pairs) >= max_train_windows:
                    return

        if isinstance(data, str):
            from ...data.diabetes_datasets.data_loader import get_loader

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
                data = getattr(loader, "processed_data", None) or getattr(
                    loader, "train_data", None
                )
            if data is None:
                raise ValueError(f"No data available from loader for {data}")

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        if isinstance(data, dict):
            # patient_id -> DataFrame
            from ...data.preprocessing.time_processing import (
                iter_daily_context_forecast_splits,
            )

            for _pid, patient_df in data.items():
                target_col = self._get_target_column(patient_df)
                input_cols = self._get_input_columns(patient_df, target_col)
                pairs_before = len(pairs)
                for daytime, nocturnal in iter_daily_context_forecast_splits(
                    patient_df
                ):
                    ctx = self._build_context_matrix(daytime, input_cols)
                    tgt = np.asarray(
                        nocturnal[target_col].values[:fcast_len],
                        dtype=np.float32,
                    )
                    if np.isnan(tgt).any():
                        continue
                    if len(ctx) < 10 or len(tgt) < fcast_len:
                        continue
                    if ctx.shape[0] > ctx_len:
                        ctx = ctx[-ctx_len:, :]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
                if len(pairs) == pairs_before and require_target:
                    _append_rolling_pairs(patient_df, target_col, input_cols)
                if require_target and len(pairs) >= max_train_windows:
                    return pairs[:max_train_windows]
        elif isinstance(data, pd.DataFrame):
            target_col = self._get_target_column(data)
            input_cols = self._get_input_columns(data, target_col)
            # Single context window (e.g. from holdout_eval): exactly ctx_len rows, no future target
            if len(data) == ctx_len:
                ctx = self._build_context_matrix(data, input_cols)
                if len(ctx) == 0:
                    return pairs  # No valid pair
                if len(ctx) < 10:
                    return pairs
                if len(ctx) > ctx_len:
                    ctx = ctx[-ctx_len:, :]
                # Dummy target for loader; only context is used in predict()
                dummy_tgt = np.zeros(fcast_len, dtype=np.float32)
                pairs.append((ctx, dummy_tgt))
                return pairs
            if not require_target:
                # Single long series: last ctx_len -> predict next fcast_len
                mat = self._build_context_matrix(data, input_cols)
                if len(mat) >= ctx_len + fcast_len:
                    ctx = mat[-ctx_len - fcast_len : -fcast_len, :]
                    tgt = mat[-fcast_len:, 0]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
            else:
                from ...data.preprocessing.time_processing import (
                    iter_daily_context_forecast_splits,
                )

                if ColumnNames.P_NUM.value in data.columns:
                    for _pid, patient_df in data.groupby(ColumnNames.P_NUM.value):
                        patient_input_cols = self._get_input_columns(
                            patient_df, target_col
                        )
                        pairs_before = len(pairs)
                        for daytime, nocturnal in iter_daily_context_forecast_splits(
                            patient_df
                        ):
                            ctx = self._build_context_matrix(
                                daytime, patient_input_cols
                            )
                            tgt = np.asarray(
                                nocturnal[target_col].values[:fcast_len],
                                dtype=np.float32,
                            )
                            if np.isnan(tgt).any():
                                continue
                            if len(ctx) < 10 or len(tgt) < fcast_len:
                                continue
                            if ctx.shape[0] > ctx_len:
                                ctx = ctx[-ctx_len:, :]
                            pairs.append(
                                (ctx.astype(np.float32), tgt.astype(np.float32))
                            )
                        if len(pairs) == pairs_before and require_target:
                            _append_rolling_pairs(
                                patient_df, target_col, patient_input_cols
                            )
                        if require_target and len(pairs) >= max_train_windows:
                            return pairs[:max_train_windows]
                else:
                    # Single patient
                    patient_input_cols = self._get_input_columns(data, target_col)
                    for daytime, nocturnal in iter_daily_context_forecast_splits(data):
                        ctx = self._build_context_matrix(daytime, patient_input_cols)
                        tgt = np.asarray(
                            nocturnal[target_col].values[:fcast_len],
                            dtype=np.float32,
                        )
                        if np.isnan(tgt).any():
                            continue
                        if len(ctx) < 10 or len(tgt) < fcast_len:
                            continue
                        if ctx.shape[0] > ctx_len:
                            ctx = ctx[-ctx_len:, :]
                        pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
                    if not pairs and require_target:
                        _append_rolling_pairs(data, target_col, patient_input_cols)
                    if require_target and len(pairs) >= max_train_windows:
                        return pairs[:max_train_windows]
        else:
            raise ValueError(
                "data must be a dataset name (str), a DataFrame, or a dict of DataFrames"
            )

        if require_target and len(pairs) > max_train_windows:
            pairs = pairs[:max_train_windows]

        return pairs

    @staticmethod
    def _normalize_split_config(
        split_config: Dict[str, float],
    ) -> Tuple[float, float, float]:
        train_ratio = float(split_config.get("train", 0.7))
        val_ratio = float(split_config.get("val", 0.2))
        test_ratio = float(split_config.get("test", 0.1))
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("split_config ratios must sum to a positive value")
        return train_ratio / total, val_ratio / total, test_ratio / total

    def _split_context_target_pairs(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
        split_config: Optional[Dict[str, float]],
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray]],
        List[Tuple[np.ndarray, np.ndarray]],
        List[Tuple[np.ndarray, np.ndarray]],
    ]:
        train_pairs = pairs
        val_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        test_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        if split_config and len(pairs) > 1:
            import random

            shuffled = list(pairs)
            random.Random(42).shuffle(shuffled)
            train_ratio, val_ratio, _ = self._normalize_split_config(split_config)
            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            train_pairs = shuffled[:n_train]
            val_pairs = shuffled[n_train : n_train + n_val]
            test_pairs = shuffled[n_train + n_val :]
            info_print(
                f"Data split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test"
            )

        return train_pairs, val_pairs, test_pairs

    @staticmethod
    def _normalize_context_array(context: np.ndarray) -> np.ndarray:
        context_arr = np.asarray(context, dtype=np.float32)
        if context_arr.ndim == 1:
            return context_arr[:, None]
        if context_arr.ndim != 2:
            raise ValueError(f"Context must be 1D or 2D, got {context_arr.shape}")
        return context_arr

    def _build_context_target_dataset(
        self, pair_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[_ContextTargetDataset]:
        if not pair_list:
            return None

        contexts = [self._normalize_context_array(p[0]) for p in pair_list]
        targets = [np.asarray(p[1], dtype=np.float32) for p in pair_list]
        context_lengths = [c.shape[0] for c in contexts]
        max_ctx = max(context_lengths)
        forecast_len = targets[0].shape[0]
        n_channels = contexts[0].shape[1]

        ctx_padded = np.zeros((len(contexts), max_ctx, n_channels), dtype=np.float32)
        tgt_stacked = np.zeros((len(targets), forecast_len), dtype=np.float32)
        for i, (context, target) in enumerate(zip(contexts, targets)):
            ctx_padded[i, -len(context) :, :] = context
            tgt_stacked[i] = target

        return _ContextTargetDataset(
            [ctx_padded[i] for i in range(len(ctx_padded))],
            [tgt_stacked[i] for i in range(len(tgt_stacked))],
            context_lengths=context_lengths,
        )

    def _build_context_target_loader(
        self,
        pair_list: List[Tuple[np.ndarray, np.ndarray]],
        *,
        batch_size: Optional[int],
        shuffle: bool,
    ) -> Optional[DataLoader]:
        dataset = self._build_context_target_dataset(pair_list)
        if dataset is None:
            return None
        resolved_batch_size = (
            batch_size if batch_size is not None else self.config.batch_size
        )
        return DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _prepare_training_data(
        self,
        train_data: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Build DataLoaders of (context, target) batches with train/val/test splitting.

        Splits data based on self.config.split_config if available, otherwise uses all data for training.
        """
        pairs = self._get_context_target_pairs(train_data, require_target=True)
        if not pairs:
            raise ValueError("No (context, target) pairs produced from train_data")

        split_config = getattr(self.config, "split_config", None)
        if split_config is None and hasattr(self.config, "data_config"):
            split_config = getattr(self.config.data_config, "split_config", None)
        train_pairs, val_pairs, test_pairs = self._split_context_target_pairs(
            pairs,
            split_config,
        )
        train_loader = self._build_context_target_loader(
            train_pairs,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = self._build_context_target_loader(
            val_pairs,
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = self._build_context_target_loader(
            test_pairs,
            batch_size=batch_size,
            shuffle=False,
        )
        if train_loader is None:
            raise ValueError("No training loader could be built from training pairs")

        return train_loader, val_loader, test_loader

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Native batched multi-episode prediction.

        Builds one context window per episode and runs batched inference in
        chunks to reduce GPU launch overhead versus per-episode predict() calls.
        """
        if quantile_levels is not None:
            raise NotImplementedError(
                "MomentForecaster does not support probabilistic forecasts."
            )
        if self.model is None:
            raise ValueError("Model not initialized")
        if episode_col not in data.columns:
            raise ValueError(
                f"Column '{episode_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        episode_ids: List[str] = []
        contexts: List[np.ndarray] = []
        context_lens: List[int] = []

        for ep_id, ep_df in data.groupby(episode_col, sort=False):
            target_col = self._get_target_column(ep_df)
            input_cols = self._get_input_columns(ep_df, target_col)
            mat = self._build_context_matrix(ep_df, input_cols)
            if len(mat) < 10:
                continue
            if len(mat) > self.config.context_length:
                mat = mat[-self.config.context_length :, :]

            episode_ids.append(str(ep_id))
            contexts.append(mat)
            context_lens.append(len(mat))

        if not contexts:
            return {}

        max_ctx = max(context_lens)
        n_channels = contexts[0].shape[1]
        ctx_padded = np.zeros((len(contexts), max_ctx, n_channels), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            ctx_padded[i, -len(ctx) :, :] = ctx

        bs = int(getattr(self.config, "batch_size", len(contexts)) or len(contexts))
        bs = max(1, bs)
        results: Dict[str, np.ndarray] = {}

        for start in range(0, len(contexts), bs):
            end = min(start + bs, len(contexts))
            batch_preds = self._forecast_batch(
                ctx_padded[start:end],
                prediction_length=self.config.forecast_length,
                context_lengths=np.asarray(context_lens[start:end], dtype=np.int64),
            )
            for i, ep_id in enumerate(episode_ids[start:end]):
                results[ep_id] = batch_preds[i]

        return results

    # --- Checkpoints ---
    def _save_checkpoint(self, output_dir: str) -> None:
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore[call-overload]
            info_print(f"Moment checkpoint saved to {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        MOMENTPipeline = _optional_moment_import()
        best_pt = os.path.join(model_dir, "best_model.pt")
        final_pt = os.path.join(model_dir, "final_model.pt")

        if os.path.isfile(best_pt) or os.path.isfile(final_pt):
            # Fine-tuned checkpoint: load base model then state dict
            if not getattr(self.config, "model_path", None):
                raise ValueError(
                    "config.model_path (base MOMENT id) required to load fine-tuned checkpoint; "
                    "ensure config.json includes model_path."
                )
            base_id = self.config.model_path
            info_print(
                f"Loading base MOMENT from {base_id}, then fine-tuned weights from {model_dir}"
            )
            model = MOMENTPipeline.from_pretrained(
                base_id,
                model_kwargs={"enable_gradient_checkpointing": True},
            )
            model = cast(Any, model)
            ckpt_path = best_pt if os.path.isfile(best_pt) else final_pt
            ckpt = torch.load(
                ckpt_path, map_location=self._device, weights_only=False
            )  # False: checkpoint may contain optimizer state with numpy scalars
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=True)
            self.model = model
        else:
            # HuggingFace-style directory
            self.model = cast(Any, MOMENTPipeline.from_pretrained(model_dir))

        model = cast(Any, self.model)
        model.init()
        model.to(self._device)
        model.eval()
        self.model = model
        self.is_fitted = True
        info_print(f"Moment checkpoint loaded from {model_dir}")

    def _save_training_config(self, output_dir: str) -> None:
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _resolve_freeze_backbone(self) -> bool:
        freeze_backbone = bool(getattr(self.config, "freeze_backbone", False))
        if hasattr(self.config, "training_config") and hasattr(
            self.config.training_config, "freeze_backbone"
        ):
            freeze_backbone = self.config.training_config.freeze_backbone
        return freeze_backbone

    def _resolve_trainable_parameters(self) -> List[torch.nn.Parameter]:
        if self.model is None:
            raise ValueError("Model not initialized")

        freeze_backbone = self._resolve_freeze_backbone()
        if not freeze_backbone:
            return list(self.model.parameters())  # type: ignore[union-attr]

        info_print("Freezing backbone parameters")
        if hasattr(self.model, "model"):
            for param in self.model.model.parameters():  # type: ignore[union-attr]
                param.requires_grad = False
        return [p for p in self.model.parameters() if p.requires_grad]  # type: ignore[union-attr]

    def _build_forecast_inputs(
        self,
        contexts: torch.Tensor,
        context_lens: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = contexts.shape[0]
        forecast_len = targets.shape[1]
        n_channels = contexts.shape[2]
        max_allowed_ctx = MOMENT_MAX_LEN - forecast_len
        if max_allowed_ctx <= 0:
            raise ValueError(
                f"forecast_length={forecast_len} must be smaller than {MOMENT_MAX_LEN}"
            )

        max_ctx = min(contexts.shape[1], max_allowed_ctx)
        contexts = contexts[:, -max_ctx:, :]
        effective_context_lens = context_lens.clamp(max=max_ctx)
        full_len = max_ctx + forecast_len

        x_enc = torch.zeros(
            batch_size,
            n_channels,
            full_len,
            dtype=torch.float32,
            device=self._device,
        )
        input_mask = torch.zeros(
            batch_size,
            full_len,
            dtype=torch.long,
            device=self._device,
        )
        normalized_targets = targets.clone()

        for i in range(batch_size):
            context_len = int(effective_context_lens[i])
            context = contexts[i, -context_len:, :].float()
            if self._use_wrapper_normalization:
                loc = context.mean(dim=0)
                scale = context.std(dim=0).clamp(min=NORMALIZATION_SCALE_FLOOR)
                context = (context - loc[None, :]) / scale[None, :]
                normalized_targets[i] = (targets[i] - loc[0]) / scale[0]
            x_enc[i, :, :context_len] = context.transpose(0, 1)
            input_mask[i, :context_len] = 1

        return x_enc, input_mask, normalized_targets, forecast_len

    def _compute_validation_loss(
        self,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
    ) -> Optional[float]:
        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_contexts = val_batch["context"].to(self._device)
                val_targets = val_batch["target"].to(self._device)
                val_context_lens = val_batch["context_len"]
                vx_enc, v_mask, val_targets_normed, forecast_len = (
                    self._build_forecast_inputs(
                        val_contexts,
                        val_context_lens,
                        val_targets,
                    )
                )
                val_output = self.model.forecast(x_enc=vx_enc, input_mask=v_mask)  # type: ignore[call-overload]
                val_preds = val_output.forecast[:, 0, -forecast_len:]  # type: ignore[union-attr]
                val_losses.append(loss_fn(val_preds, val_targets_normed).item())
        return float(np.mean(val_losses)) if val_losses else None

    # --- Training ---
    def _train_model(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fine-tune MOMENT model on training data.

        Uses masked reconstruction training: given context + target, masks the target
        portion and trains the model to reconstruct it.

        Args:
            train_data: Training data (dataset name, DataFrame, or dict of DataFrames)
            output_dir: Directory for saving checkpoints and logs
            **kwargs: Additional arguments (e.g., resume_from_checkpoint)

        Returns:
            Dictionary containing train_metrics and training_history
        """
        # Check training_mode for zero-shot (TTM pattern)
        if (
            hasattr(self.config, "training_mode")
            and self.config.training_mode == "zero_shot"
        ):
            info_print("Moment zero-shot mode: skipping training")
            return {"train_metrics": {}, "training_history": {}}

        if self.model is None:
            raise ValueError("Model not initialized")

        info_print("Starting MOMENT fine-tuning...")
        os.makedirs(output_dir, exist_ok=True)

        # Save config so holdout_eval can load checkpoint (model_path, context_length, etc.)
        self._save_training_config(output_dir)

        # Prepare data loaders
        train_loader, val_loader, _ = self._prepare_training_data(train_data)

        train_dataset = cast(_ContextTargetDataset, train_loader.dataset)
        if len(train_dataset) == 0:
            raise ValueError("No training data available")

        info_print(f"Training samples: {len(train_dataset)}")
        if val_loader:
            val_dataset = cast(_ContextTargetDataset, val_loader.dataset)
            info_print(f"Validation samples: {len(val_dataset)}")
        if self._use_wrapper_normalization:
            info_print("Using wrapper-side per-window normalization during training")
        else:
            info_print(
                "Using MOMENT internal normalization (wrapper normalization disabled)"
            )

        # Set model to training mode
        self.model.train()

        # Setup optimizer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        trainable_params = self._resolve_trainable_parameters()

        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, "weight_decay", 0.01),
        )

        num_epochs = self.config.num_epochs
        total_steps = len(train_loader) * num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Loss function
        loss_fn = torch.nn.MSELoss()

        # Training loop
        training_history = []
        best_val_loss = float("inf")
        best_model_state = None
        avg_train_loss = 0.0
        val_loss: Optional[float] = None

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                contexts = batch["context"].to(self._device)  # [B, max_ctx_len, C]
                targets = batch["target"].to(self._device)  # [B, forecast_len]
                context_lens = batch["context_len"]  # [B]

                optimizer.zero_grad()
                x_enc, input_mask, targets_normed, forecast_len = (
                    self._build_forecast_inputs(
                        contexts,
                        context_lens,
                        targets,
                    )
                )
                output = self.model.forecast(x_enc=x_enc, input_mask=input_mask)  # type: ignore[call-overload]
                # output.forecast: [B, C, full_len] — take channel 0, last forecast_len steps
                preds = output.forecast[:, 0, -forecast_len:]  # type: ignore[union-attr] # [B, forecast_len]
                batch_loss = loss_fn(preds, targets_normed)

                # Backward pass
                batch_loss.backward()

                # Gradient clipping
                grad_clip = getattr(self.config, "gradient_clip_val", 1.0)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                optimizer.step()
                scheduler.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    info_print(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {batch_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._compute_validation_loss(val_loader, loss_fn)

            # Logging
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            training_history.append(log_entry)

            info_print(
                f"Epoch {epoch + 1}/{num_epochs} completed - "
                f"Train Loss: {avg_train_loss:.6f}"
                + (f", Val Loss: {val_loss:.6f}" if val_loss is not None else "")
            )

            # Save best model
            current_loss = val_loss if val_loss is not None else avg_train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_model_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "loss": current_loss,
                }
                info_print(f"New best model (loss: {best_val_loss:.6f})")

        # Save final model
        if best_model_state:
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(best_model_state, checkpoint_path)
            info_print(f"Saved best model to {checkpoint_path}")

        # Save final checkpoint
        final_checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "config": self.config.to_dict(),
        }
        final_path = os.path.join(output_dir, "final_model.pt")
        torch.save(final_checkpoint, final_path)
        info_print(f"Saved final model to {final_path}")

        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        self.is_fitted = True
        self.model.eval()

        return {
            "train_metrics": {
                "final_train_loss": avg_train_loss,
                "best_val_loss": best_val_loss if val_loss is not None else None,
            },
            "training_history": training_history,
        }
