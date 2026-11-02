# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""TimesFM 3.0 model implementation using the base TSFM framework.

Backend/runtime role:
    Standalone adapter for Google's TimesFM 3.0 (native ``timesfm3`` package),
    providing zero-shot probabilistic inference with native past-only covariates
    and parameter-efficient LoRA fine-tuning (via ``peft``). This is a SEPARATE
    family from the published ``src/models/timesfm`` adapter and must not share
    code or checkpoints with it.

Key behavior differences:
    - Inference uses ``TimesFM3Forecaster.predict_batch`` with
      ``past_only_covariates`` built from the context window (matching the
      repository's past-only covariate convention, e.g. IOB).
    - Fine-tuning injects LoRA adapters and optimizes a pinball (quantile) loss
      through the model's grad-enabled ``decode`` path, with validation-based
      early stopping. Only the LoRA adapter weights are persisted.

Invariants/limitations:
    - Target series are read from the ``bg_mM`` column in mmol/L (NOT mg/dL).
    - The pretrained quantile head emits a fixed decile grid; arbitrary
      requested quantile levels are linearly interpolated across that grid.
    - TimesFM 3.0 covariate kernels dispatch through an optional Triton ``bmm``
      override. On hosts lacking CUDA/Python build headers, set the environment
      variable ``TORCH_DISABLE_NATIVE_JIT=1`` (before importing torch) to fall
      back to the standard ``aten`` implementation. This module sets it via
      ``setdefault`` as a best-effort default.
"""

import os

# Best-effort: disable torch's native (Triton-compiled) op overrides so the
# TimesFM 3.0 covariate `bmm` path falls back to standard aten kernels on hosts
# without CUDA/Python build headers. Must be set before torch is imported to
# take effect; harmless when torch is already initialized. See module docstring.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import json  # noqa: E402
import logging  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ...utils.logging_helper import error_print, info_print  # noqa: E402
from ..base import BaseTimeSeriesFoundationModel, TrainingBackend  # noqa: E402
from ..base.registry import ModelRegistry  # noqa: E402
from .config import TimesFM3Config  # noqa: E402

logger = logging.getLogger(__name__)

# Candidate columns used to split a flat panel DataFrame into per-series frames
# when building training windows.
_SERIES_GROUP_COLUMNS: Tuple[str, ...] = ("episode_id", "patient_id", "p_num")

# Checkpoint filename for the persisted LoRA adapter state dict.
_LORA_ADAPTER_FILENAME = "lora_adapter.pt"


def _optional_timesfm3_import():
    """Import the native TimesFM 3.0 forecaster class if ``timesfm3`` is present."""
    try:
        from timesfm3.torch.timesfm3_forecaster import (  # pyright: ignore[reportMissingImports]
            TimesFM3Forecaster as _Backend,
        )

        return _Backend
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "TimesFM 3.0 model requires the 'timesfm3' package. "
            "Install it into the .venvs/timesfm3 environment."
        ) from exc


def _optional_peft_import():
    """Import the ``peft`` helpers used for LoRA injection and checkpoint I/O."""
    try:
        from peft import (  # pyright: ignore[reportMissingImports]
            LoraConfig,
            get_peft_model_state_dict,
            inject_adapter_in_model,
            set_peft_model_state_dict,
        )

        return (
            LoraConfig,
            inject_adapter_in_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "TimesFM 3.0 fine-tuning requires the 'peft' package. "
            "Install it into the .venvs/timesfm3 environment."
        ) from exc


@ModelRegistry.register("timesfm3")
class TimesFM3Forecaster(BaseTimeSeriesFoundationModel):
    """TimesFM 3.0 forecaster with zero-shot covariates and LoRA fine-tuning.

    Public behavior conforms to :class:`BaseTimeSeriesFoundationModel`:
    ``predict``/``predict_batch``/``fit``/``save``/``load`` are inherited and
    delegate to the protected hooks implemented here. Probabilistic forecasts
    are produced by interpolating the pretrained decile quantile head.

    Contract obligations:
        - ``supports_zero_shot`` is True (usable without fitting).
        - ``supports_probabilistic_forecast`` is True (accepts ``quantile_levels``).
        - Fine-tuning persists only the LoRA adapter; the base weights come from
          ``config.model_path``.
    """

    config_class = TimesFM3Config
    config: TimesFM3Config  # Override base class typing

    def __init__(self, config: TimesFM3Config):
        """Initialize the forecaster and load the pretrained TimesFM 3.0 weights.

        Args:
            config: TimesFM 3.0 configuration object.
        """
        # Runtime handles populated by _initialize_model (called by super().__init__).
        self.forecaster: Optional[Any] = None
        self.device: str = "cpu"
        self._model_quantiles: List[float] = []
        self._lora_injected: bool = False
        self._rng = np.random.default_rng(0)
        super().__init__(config)

    # --- Required properties ---
    @property
    def training_backend(self) -> TrainingBackend:
        """Return the training backend (custom PyTorch LoRA loop)."""
        return TrainingBackend.PYTORCH

    @property
    def supports_zero_shot(self) -> bool:
        """Return whether this model supports zero-shot inference."""
        return True

    @property
    def supports_probabilistic_forecast(self) -> bool:
        """Return whether this model supports quantile forecasts."""
        return True

    # --- Lifecycle / initialization ---
    def _initialize_model(self) -> None:
        """Load the pretrained TimesFM 3.0 forecaster onto the target device."""
        backend_cls = _optional_timesfm3_import()

        use_cuda = torch.cuda.is_available() and not getattr(
            self.config, "use_cpu", False
        )
        device = self.config.device or ("cuda" if use_cuda else "cpu")
        self.device = device

        info_print(
            f"Initializing TimesFM 3.0 from {self.config.model_path} on {device}..."
        )
        forecaster = backend_cls.from_pretrained(self.config.model_path, device=device)
        if forecaster is None or getattr(forecaster, "model", None) is None:
            error_print("Failed to load TimesFM 3.0 forecaster.")
            raise ValueError("TimesFM 3.0 model loading failed.")

        self.forecaster = forecaster
        self.model = forecaster.model
        self._model_quantiles = list(forecaster.model.quantiles)
        self._lora_injected = False
        self.model.eval()
        info_print(
            f"TimesFM 3.0 initialized on {device} (quantiles={self._model_quantiles})"
        )

    # --- Training path ---
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[
        List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]], Optional[list], None
    ]:
        """Build sliding training windows and split them for early stopping.

        Args:
            train_data: DataFrame, dict/list of DataFrames, or a dataset name.

        Returns:
            Tuple ``(train_windows, val_windows, None)`` where each window is
            ``(context_target[ctx], covariates[C, ctx] | None, future_target[H])``.
            ``val_windows`` is None when the validation split is disabled or the
            window count is too small.
        """
        windows = self._build_windows(train_data)
        if not windows:
            raise ValueError(
                "No training windows could be built from train_data "
                f"(need >= context_length + forecast_length = "
                f"{self.config.context_length + self.config.forecast_length} rows per series)."
            )

        val_fraction = float(getattr(self.config, "val_fraction", 0.0) or 0.0)
        if val_fraction > 0.0 and len(windows) >= 4:
            order = self._rng.permutation(len(windows))
            n_val = max(1, int(round(len(windows) * val_fraction)))
            n_val = min(n_val, len(windows) - 1)
            val_windows = [windows[i] for i in order[:n_val]]
            train_windows = [windows[i] for i in order[n_val:]]
        else:
            train_windows, val_windows = windows, None

        return train_windows, val_windows, None

    def _train_model(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fine-tune LoRA adapters with a pinball loss and early stopping.

        Args:
            train_data: Training data (DataFrame, dict/list of DataFrames, or
                dataset name).
            output_dir: Directory where the adapter checkpoint and config are saved.
            **kwargs: Unused; accepted for base-class compatibility.

        Returns:
            Dict with ``train_metrics`` and ``training_history``.
        """
        del kwargs
        if getattr(self.config, "training_mode", "fine_tune") == "zero_shot":
            info_print("TimesFM 3.0 zero-shot mode: skipping training")
            return {"train_metrics": {}, "training_history": {}}

        if self.forecaster is None or self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        os.makedirs(output_dir, exist_ok=True)
        self._write_config_json(output_dir)

        train_windows, val_windows, _ = self._prepare_training_data(train_data)
        self._inject_lora()

        decode_fn = getattr(
            type(self.model).decode, "__wrapped__", type(self.model).decode
        )
        _, _, get_state_dict, set_state_dict = _optional_peft_import()

        train_tensors = self._windows_to_tensors(train_windows)
        val_tensors = self._windows_to_tensors(val_windows) if val_windows else None
        info_print(
            f"TimesFM 3.0 fine-tuning: {len(train_windows)} train windows"
            + (f", {len(val_windows)} val windows" if val_windows else "")
        )

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, "weight_decay", 0.01),
        )

        num_epochs = int(self.config.num_epochs)
        patience = int(getattr(self.config, "early_stopping_patience", 0) or 0)
        best_monitor = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_without_improve = 0
        history: List[Dict[str, float]] = []
        final_train_loss = float("nan")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self._run_epoch(
                train_tensors, decode_fn, optimizer, trainable, train=True
            )
            final_train_loss = train_loss

            val_loss: Optional[float] = None
            if val_tensors is not None:
                self.model.eval()
                val_loss = self._run_epoch(
                    val_tensors, decode_fn, None, trainable, train=False
                )

            entry: Dict[str, float] = {"epoch": epoch + 1, "train_loss": train_loss}
            if val_loss is not None:
                entry["val_loss"] = val_loss
            history.append(entry)
            info_print(
                f"Epoch {epoch + 1}/{num_epochs} - train_pinball={train_loss:.4f}"
                + (f", val_pinball={val_loss:.4f}" if val_loss is not None else "")
            )

            monitor = val_loss if val_loss is not None else train_loss
            if np.isfinite(monitor) and monitor < best_monitor - 1e-6:
                best_monitor = monitor
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in get_state_dict(self.model).items()
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if patience and epochs_without_improve >= patience:
                    info_print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best monitor={best_monitor:.4f})"
                    )
                    break

        if best_state is not None:
            set_state_dict(self.model, best_state)
        self.model.eval()
        self.is_fitted = True

        self._save_checkpoint(output_dir)

        return {
            "train_metrics": {
                "final_train_loss": final_train_loss,
                "best_val_loss": best_monitor if np.isfinite(best_monitor) else None,
            },
            "training_history": history,
        }

    # --- Inference path ---
    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Forecast a single context window.

        Args:
            data: DataFrame with the target column (``config.target_col``, e.g.
                ``bg_mM`` in mmol/L) and optional past-only covariate columns.
            quantile_levels: When set, return quantile trajectories of shape
                ``(len(quantile_levels), forecast_length)``; otherwise return a
                point forecast of shape ``(forecast_length,)``.
            **kwargs: Unused.

        Returns:
            Numpy forecast array (see ``quantile_levels`` for shape).
        """
        del kwargs
        if self.forecaster is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        horizon = int(self.config.forecast_length)
        context = self._extract_context(data)
        past_only = self._extract_covariates(data, len(context))

        want_quantiles = quantile_levels is not None
        output = next(
            iter(
                self.forecaster.predict_batch(
                    [context],
                    horizon=horizon,
                    past_only_covariates=[past_only] if past_only is not None else None,
                    return_quantiles=want_quantiles,
                    make_positive=self.config.make_positive,
                )
            )
        )

        if want_quantiles:
            return self._interpolate_quantiles(
                output.quantiles, quantile_levels, horizon
            )
        return np.asarray(output.forecast, dtype=np.float64).ravel()[:horizon]

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Forecast multiple episodes using TimesFM 3.0 native batching.

        Args:
            data: Panel DataFrame with an ``episode_col`` column.
            episode_col: Column identifying episodes.
            quantile_levels: When set, per-episode quantile trajectories.

        Returns:
            Dict mapping episode id (str) to a forecast array.
        """
        if self.forecaster is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        horizon = int(self.config.forecast_length)
        contexts: List[np.ndarray] = []
        covariates: List[Optional[np.ndarray]] = []
        episode_ids: List[str] = []
        any_covariates = False

        for episode_id, episode_df in data.groupby(episode_col, sort=False):
            context = self._extract_context(episode_df)
            past_only = self._extract_covariates(episode_df, len(context))
            contexts.append(context)
            covariates.append(past_only)
            any_covariates = any_covariates or past_only is not None
            episode_ids.append(str(episode_id))

        if not contexts:
            return {}

        want_quantiles = quantile_levels is not None
        outputs = list(
            self.forecaster.predict_batch(
                contexts,
                horizon=horizon,
                past_only_covariates=covariates if any_covariates else None,
                return_quantiles=want_quantiles,
                make_positive=self.config.make_positive,
            )
        )

        results: Dict[str, np.ndarray] = {}
        for episode_id, output in zip(episode_ids, outputs):
            if want_quantiles:
                results[episode_id] = self._interpolate_quantiles(
                    output.quantiles, quantile_levels, horizon
                )
            else:
                results[episode_id] = np.asarray(
                    output.forecast, dtype=np.float64
                ).ravel()[:horizon]
        return results

    # --- Serialization ---
    def _save_checkpoint(self, output_dir: str) -> None:
        """Persist only the LoRA adapter weights (no-op in zero-shot mode)."""
        if not self._lora_injected or self.model is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        _, _, get_state_dict, _ = _optional_peft_import()
        state = get_state_dict(self.model)
        torch.save(state, os.path.join(output_dir, _LORA_ADAPTER_FILENAME))
        info_print(f"TimesFM 3.0 LoRA adapter saved to {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load a saved LoRA adapter into the freshly initialized base model."""
        adapter_path = os.path.join(model_dir, _LORA_ADAPTER_FILENAME)
        if not os.path.isfile(adapter_path):
            info_print(
                f"No LoRA adapter found in {model_dir}; using zero-shot weights."
            )
            return
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        self._inject_lora()
        _, _, _, set_state_dict = _optional_peft_import()
        state = torch.load(adapter_path, map_location=self.device)
        set_state_dict(self.model, state)
        self.model.eval()
        self.is_fitted = True
        info_print(f"TimesFM 3.0 LoRA adapter loaded from {model_dir}")

    # --- Private helpers ---
    def _inject_lora(self) -> None:
        """Inject LoRA adapters once and mark only adapter params trainable."""
        if self._lora_injected:
            return
        lora_config_cls, inject_adapter, _, _ = _optional_peft_import()
        lora_config = lora_config_cls(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        injected = inject_adapter(lora_config, self.model)
        if injected is not None:
            self.model = injected
            self.forecaster.model = injected
        for name, param in self.model.named_parameters():
            param.requires_grad_("lora_" in name)
        self._lora_injected = True

    def _pinball_loss(
        self, logits_q: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean pinball (quantile) loss over the model's quantile grid.

        Args:
            logits_q: Predicted quantiles of shape ``(b, H, Q)``.
            target: Ground-truth horizon of shape ``(b, H)``.
        """
        quantiles = torch.tensor(
            self._model_quantiles, device=logits_q.device, dtype=logits_q.dtype
        ).view(1, 1, -1)
        errors = target.unsqueeze(-1) - logits_q
        return torch.maximum(quantiles * errors, (quantiles - 1) * errors).mean()

    def _run_epoch(
        self,
        tensors: Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
        decode_fn: Any,
        optimizer: Optional[torch.optim.Optimizer],
        trainable: List[torch.nn.Parameter],
        train: bool,
    ) -> float:
        """Run one train or validation epoch and return the mean pinball loss."""
        cgm, cov, true = tensors
        horizon = int(self.config.forecast_length)
        n_windows = cgm.shape[0]
        batch_size = max(1, int(self.config.batch_size))
        order = self._rng.permutation(n_windows) if train else np.arange(n_windows)
        jitter = float(getattr(self.config, "covariate_jitter", 0.0) or 0.0)
        grad_clip = float(getattr(self.config, "grad_clip_norm", 1.0) or 0.0)

        total_loss, num_batches = 0.0, 0
        for start in range(0, n_windows, batch_size):
            idx = order[start : start + batch_size]
            target = cgm[idx].unsqueeze(1).to(self.device)  # (b, 1, ctx)
            y_true = true[idx].to(self.device)  # (b, H)
            past_only: Optional[torch.Tensor] = None
            if cov is not None:
                past_only = cov[idx].to(self.device)  # (b, C, ctx)
                if train and jitter > 0.0:
                    past_only = past_only + jitter * torch.randn_like(past_only)

            if train:
                logits = decode_fn(
                    self.model,
                    target=target,
                    past_only_covariates=past_only,
                    horizon=horizon,
                )
            else:
                with torch.no_grad():
                    logits = decode_fn(
                        self.model,
                        target=target,
                        past_only_covariates=past_only,
                        horizon=horizon,
                    )
            # logits: (b, num_variates, H, Q); the target variate is index 0.
            loss = self._pinball_loss(logits[:, 0, :horizon, :], y_true)

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                grads_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in trainable
                )
                if not grads_finite:
                    optimizer.zero_grad()
                    continue
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / num_batches if num_batches else float("nan")

    def _windows_to_tensors(
        self,
        windows: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Stack a list of windows into (context, covariates, future) tensors."""
        cgm = torch.tensor(np.stack([w[0] for w in windows]), dtype=torch.float32)
        true = torch.tensor(np.stack([w[2] for w in windows]), dtype=torch.float32)
        cov: Optional[torch.Tensor] = None
        if windows[0][1] is not None:
            cov = torch.tensor(
                np.stack([w[1] for w in windows]),
                dtype=torch.float32,  # type: ignore[misc]
            )
        return cgm, cov, true

    def _build_windows(
        self, data: Any
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        """Build sliding ``(context, covariates, future)`` windows from data."""
        ctx_len = int(self.config.context_length)
        horizon = int(self.config.forecast_length)
        stride = int(self.config.window_stride or horizon)
        stride = max(1, stride)
        max_windows = int(getattr(self.config, "max_train_windows", 2048))
        target_col = self.config.target_col

        windows: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]] = []
        for series in self._iter_series(data):
            if target_col not in series.columns:
                continue
            target = series[target_col].to_numpy(dtype=np.float64)
            cov_cols = [c for c in self.config.covariate_cols if c in series.columns]
            cov_matrix = (
                np.nan_to_num(series[cov_cols].to_numpy(dtype=np.float64)).T
                if cov_cols
                else None
            )  # (C, T)

            n_rows = target.shape[0]
            if n_rows < ctx_len + horizon:
                continue
            last_start = n_rows - (ctx_len + horizon) + 1
            for i in range(0, last_start, stride):
                context = target[i : i + ctx_len]
                future = target[i + ctx_len : i + ctx_len + horizon]
                if not np.isfinite(context).all() or not np.isfinite(future).all():
                    continue
                cov_window = (
                    cov_matrix[:, i : i + ctx_len].astype(np.float32)
                    if cov_matrix is not None
                    else None
                )
                windows.append(
                    (context.astype(np.float32), cov_window, future.astype(np.float32))
                )
                if len(windows) >= max_windows:
                    return windows
        return windows

    def _iter_series(self, data: Any):
        """Yield per-series DataFrames from a DataFrame, dict/list, or dataset name."""
        if isinstance(data, str):
            from ...data.diabetes_datasets.data_loader import get_loader

            loader = get_loader(  # type: ignore[call-overload]
                data_source_name=data,  # type: ignore[arg-type]
                dataset_type="train",
                use_cached=True,
            )
            resolved = (
                getattr(loader, "validation_data", None)
                or getattr(loader, "processed_data", None)
                or getattr(loader, "train_data", None)
            )
            if resolved is None:
                raise ValueError(f"No data available from loader for '{data}'")
            data = resolved

        if isinstance(data, dict):
            for frame in data.values():
                yield frame
        elif isinstance(data, (list, tuple)):
            for frame in data:
                yield frame
        elif isinstance(data, pd.DataFrame):
            group_col = next(
                (c for c in _SERIES_GROUP_COLUMNS if c in data.columns), None
            )
            if group_col is not None:
                for _, frame in data.groupby(group_col, sort=False):
                    yield frame
            else:
                yield data
        else:
            raise ValueError(
                "train_data must be a DataFrame, dict/list of DataFrames, or a "
                f"dataset name (str); got {type(data)!r}"
            )

    def _extract_context(self, data: pd.DataFrame) -> np.ndarray:
        """Return the trailing context-length target values as float32."""
        target_col = self.config.target_col
        if target_col not in data.columns:
            raise ValueError(
                f"DataFrame must contain the target column '{target_col}'. "
                f"Available columns: {list(data.columns)}"
            )
        ctx_len = int(self.config.context_length)
        context = data[target_col].to_numpy(dtype=np.float64)[-ctx_len:]
        return context.astype(np.float32)

    def _extract_covariates(
        self, data: pd.DataFrame, context_len: int
    ) -> Optional[np.ndarray]:
        """Return past-only covariates of shape ``(C, context_len)`` or None."""
        cov_cols = [c for c in self.config.covariate_cols if c in data.columns]
        if not cov_cols:
            return None
        matrix = data[cov_cols].to_numpy(dtype=np.float64)[-context_len:]
        matrix = np.nan_to_num(matrix).T  # (C, context_len)
        return matrix.astype(np.float32)

    def _interpolate_quantiles(
        self,
        model_quantiles: Any,
        requested_levels: List[float],
        horizon: int,
    ) -> np.ndarray:
        """Interpolate the fixed decile grid onto requested quantile levels.

        Args:
            model_quantiles: Model output of shape ``(H, Q)``.
            requested_levels: Desired quantile levels in (0, 1).
            horizon: Forecast horizon (H).

        Returns:
            Array of shape ``(len(requested_levels), horizon)``.
        """
        grid = np.asarray(self._model_quantiles, dtype=np.float64)
        values = np.asarray(model_quantiles, dtype=np.float64)[:horizon]  # (H, Q)
        levels = np.asarray(requested_levels, dtype=np.float64)
        result = np.empty((levels.shape[0], values.shape[0]), dtype=np.float64)
        for h in range(values.shape[0]):
            result[:, h] = np.interp(levels, grid, values[h])
        return result

    def _write_config_json(self, output_dir: str) -> None:
        """Write ``config.json`` so a fine-tuned checkpoint can be reloaded."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)
