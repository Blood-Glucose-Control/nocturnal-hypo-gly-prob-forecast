# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Shared base class for Darts-backed forecasting models.

This mirrors the AutoGluon base pattern: model-specific wrappers stay thin while
shared data preparation, training, inference, and checkpoint logic lives here.
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

import numpy as np
import pandas as pd

from ..data.preprocessing.gap_handling import segment_all_patients
from ..utils.logging_helper import info_print
from .base import BaseTimeSeriesFoundationModel, TrainingBackend
from .tide.utils import convert_to_patient_dict

logger = logging.getLogger(__name__)


class DartsRuntimeConfig(Protocol):
    """Config contract required by DartsGlobalModelBase."""

    context_length: int
    forecast_length: int
    batch_size: int
    num_epochs: int
    covariate_cols: List[str]
    time_col: str
    target_col: str
    interval_mins: int
    patient_col: str
    min_segment_length: Optional[int]
    imputation_threshold_mins: int


class DartsForecast(Protocol):
    """Minimal forecast contract returned by Darts predict()."""

    @property
    def components(self) -> Any: ...

    def values(self, copy: bool = ...) -> Any: ...


class DartsModelContract(Protocol):
    """Minimal Darts model contract used by this base wrapper."""

    def fit(self, **kwargs: Any) -> Any: ...

    def predict(self, **kwargs: Any) -> DartsForecast: ...

    def save(self, path: str) -> None: ...


class DartsGlobalModelBase(BaseTimeSeriesFoundationModel):
    """Shared base for Darts global forecasting models."""

    _DARTS_MODEL_FILENAME = "darts_model.pkl"

    @property
    def _cfg(self) -> DartsRuntimeConfig:
        return cast(DartsRuntimeConfig, self.config)

    @property
    def _darts_model(self) -> DartsModelContract:
        if self.model is None:
            raise ValueError("Model must be initialized before use.")
        return cast(DartsModelContract, self.model)

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_zero_shot(self) -> bool:
        return False

    def _initialize_model(self) -> None:
        """No-op: Darts model is created lazily in _train_model()."""
        pass

    @abstractmethod
    def _create_darts_model(self) -> Any:
        """Create and return the concrete Darts model instance."""

    @abstractmethod
    def _load_darts_model(self, model_path: str) -> Any:
        """Load and return the concrete Darts model from disk."""

    def _train_model_info_log(self) -> None:
        config = self._cfg
        cov_str = (
            f", covariates: {config.covariate_cols}" if config.covariate_cols else ""
        )
        info_print(
            f"Starting {self.__class__.__name__} training: "
            f"context={config.context_length}, forecast={config.forecast_length}, "
            f"batch={config.batch_size}, epochs={config.num_epochs}{cov_str}"
        )

    def _to_target_and_covariates(
        self, data: pd.DataFrame
    ) -> Tuple[Any, Optional[Any]]:
        """Convert a per-episode/segment DataFrame into Darts TimeSeries objects."""
        from darts import TimeSeries  # type: ignore[import-not-found]

        config = self._cfg
        if config.time_col in data.columns:
            frame = data.copy()
            frame[config.time_col] = pd.to_datetime(frame[config.time_col])
            frame = frame.sort_values(config.time_col)
        elif isinstance(data.index, pd.DatetimeIndex):
            frame = data.copy().sort_index().reset_index()
            frame = frame.rename(columns={frame.columns[0]: config.time_col})
        else:
            raise ValueError(
                f"Expected '{config.time_col}' column or DatetimeIndex in input "
                f"data, got columns={list(data.columns)}"
            )

        if config.target_col not in frame.columns:
            raise ValueError(
                f"Target column '{config.target_col}' not found. "
                f"Available columns: {list(frame.columns)}"
            )

        freq = f"{config.interval_mins}min"
        target_series = TimeSeries.from_dataframe(
            frame,
            time_col=config.time_col,
            value_cols=[config.target_col],
            fill_missing_dates=False,
            freq=freq,
        )

        if not config.covariate_cols:
            return target_series, None

        cov_frame = frame[[config.time_col]].copy()
        for cov_col in config.covariate_cols:
            if cov_col not in frame.columns:
                logger.warning(
                    "Covariate '%s' missing from input; filling with zeros.", cov_col
                )
                cov_frame[cov_col] = 0.0
            else:
                cov_frame[cov_col] = frame[cov_col].ffill().fillna(0.0)

        past_covariates = TimeSeries.from_dataframe(
            cov_frame,
            time_col=config.time_col,
            value_cols=list(config.covariate_cols),
            fill_missing_dates=False,
            freq=freq,
        )
        return target_series, past_covariates

    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[List[Any], Optional[List[Any]], Optional[Any]]:
        """Prepare segmented per-series Darts training inputs."""
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} expects train_data as pandas DataFrame, "
                f"got {type(train_data).__name__}"
            )

        config = self._cfg
        patient_dict = convert_to_patient_dict(
            train_data,
            patient_col=config.patient_col,
            time_col=config.time_col,
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")
        if not segments:
            raise ValueError(
                "No usable segments after gap handling. "
                "Check imputation_threshold_mins/min_segment_length."
            )

        train_series: List[Any] = []
        covariate_series: List[Any] = []
        expected_delta = pd.Timedelta(minutes=config.interval_mins)
        split_segments = 0
        dropped_short_chunks = 0
        for segment_key, segment_df in segments.items():
            contiguous_chunks = self._split_segment_on_time_gaps(
                segment_df=segment_df,
                expected_delta=expected_delta,
                min_chunk_length=config.min_segment_length,
            )
            split_segments += len(contiguous_chunks)
            if not contiguous_chunks:
                dropped_short_chunks += 1
                logger.warning(
                    "Dropped segment %s after time-gap splitting: no chunk met "
                    "min_segment_length=%s",
                    segment_key,
                    config.min_segment_length,
                )
                continue

            for chunk_df in contiguous_chunks:
                target_ts, cov_ts = self._to_target_and_covariates(chunk_df)
                target_values = np.asarray(target_ts.values(copy=False))
                if not np.isfinite(target_values).all():
                    raise ValueError(
                        "Non-finite target values detected in Darts training series "
                        f"for segment {segment_key}."
                    )
                train_series.append(target_ts)
                if cov_ts is not None:
                    cov_values = np.asarray(cov_ts.values(copy=False))
                    if not np.isfinite(cov_values).all():
                        raise ValueError(
                            "Non-finite covariate values detected in Darts training "
                            f"series for segment {segment_key}."
                        )
                    covariate_series.append(cov_ts)

        if not train_series:
            raise ValueError(
                "No trainable series produced from segmented training data."
            )

        if split_segments != len(segments):
            info_print(
                f"Time-gap split expanded {len(segments)} segments into "
                f"{split_segments} contiguous chunks "
                f"(dropped: {dropped_short_chunks})"
            )
        info_print(f"Prepared {len(train_series)} Darts training series")
        return train_series, (covariate_series or None), None

    def _split_segment_on_time_gaps(
        self,
        segment_df: pd.DataFrame,
        expected_delta: pd.Timedelta,
        min_chunk_length: Optional[int],
    ) -> List[pd.DataFrame]:
        """Split a segment into contiguous chunks where timestamp deltas stay regular."""
        if segment_df.empty:
            return []

        config = self._cfg
        if isinstance(segment_df.index, pd.DatetimeIndex):
            ordered = segment_df.sort_index()
            ordered = ordered[~ordered.index.duplicated(keep="last")]
            deltas = ordered.index.to_series().diff()
            gap_breaks = deltas.notna() & (deltas != expected_delta)
            group_ids = gap_breaks.cumsum()
            chunks = [chunk for _, chunk in ordered.groupby(group_ids)]
        elif config.time_col in segment_df.columns:
            ordered = segment_df.copy()
            ordered[config.time_col] = pd.to_datetime(ordered[config.time_col])
            ordered = ordered.sort_values(config.time_col)
            ordered = ordered.drop_duplicates(subset=[config.time_col], keep="last")
            deltas = ordered[config.time_col].diff()
            gap_breaks = deltas.notna() & (deltas != expected_delta)
            group_ids = gap_breaks.cumsum()
            chunks = [chunk for _, chunk in ordered.groupby(group_ids)]
        else:
            raise ValueError(
                f"Expected DatetimeIndex or '{config.time_col}' column in segment."
            )

        if min_chunk_length is None:
            return chunks
        return [chunk for chunk in chunks if len(chunk) >= min_chunk_length]

    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        del output_dir, kwargs

        train_series, covariate_series, _ = self._prepare_training_data(train_data)
        self.model = self._create_darts_model()
        if self.model is None:
            raise RuntimeError("Failed to create Darts model instance.")

        fit_kwargs: Dict[str, Any] = {"series": train_series, "verbose": False}
        if covariate_series is not None:
            fit_kwargs["past_covariates"] = covariate_series

        self._train_model_info_log()
        self._darts_model.fit(**fit_kwargs)

        return {
            "train_metrics": {
                "status": "completed",
                "num_series": len(train_series),
                "uses_past_covariates": covariate_series is not None,
            }
        }

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        del kwargs
        if self.model is None:
            raise ValueError("Model must be fitted or loaded before prediction.")

        config = self._cfg
        target_series, past_covariates = self._to_target_and_covariates(data)
        predict_kwargs: Dict[str, Any] = {
            "n": config.forecast_length,
            "series": target_series,
            "verbose": False,
        }
        if past_covariates is not None:
            predict_kwargs["past_covariates"] = past_covariates

        if quantile_levels is None:
            forecast = self._darts_model.predict(**predict_kwargs)
            values = np.asarray(forecast.values(copy=False))
            if values.ndim == 1:
                return values.astype(float)
            if values.ndim == 2 and values.shape[1] == 1:
                return values[:, 0].astype(float)
            raise ValueError(
                f"Unexpected forecast shape {values.shape}; expected "
                f"(forecast_length,) or (forecast_length, 1)."
            )

        resolved_quantiles = [float(q) for q in quantile_levels]
        predict_kwargs["predict_likelihood_parameters"] = True
        forecast = self._darts_model.predict(**predict_kwargs)
        quantile_values = np.asarray(forecast.values(copy=False))
        components = [str(component) for component in forecast.components]

        if quantile_values.ndim != 2:
            raise ValueError(
                "Unexpected quantile forecast tensor shape "
                f"{quantile_values.shape}; expected (forecast_length, n_quantiles)."
            )

        quantile_column_lookup: Dict[float, int] = {}
        for idx, component_name in enumerate(components):
            if "_q" not in component_name:
                continue
            try:
                q = round(float(component_name.rsplit("_q", 1)[1]), 6)
            except ValueError:
                continue
            quantile_column_lookup[q] = idx

        if not quantile_column_lookup:
            raise ValueError(
                "Quantile forecast requested, but likelihood-parameter columns were "
                "not returned by the Darts model."
            )

        output: List[np.ndarray] = []
        for requested_quantile in resolved_quantiles:
            key = round(requested_quantile, 6)
            if key not in quantile_column_lookup:
                available = sorted(quantile_column_lookup.keys())
                raise ValueError(
                    f"Requested quantile {requested_quantile} not available. "
                    f"Available quantiles: {available}"
                )
            output.append(quantile_values[:, quantile_column_lookup[key]].astype(float))

        return np.vstack(output)

    def _save_checkpoint(self, output_dir: str) -> None:
        if self.model is None:
            raise ValueError(
                "Cannot save checkpoint before model initialization/training."
            )
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, self._DARTS_MODEL_FILENAME)
        self._darts_model.save(model_path)
        logger.info("Darts model checkpoint saved to %s", model_path)

    def _load_checkpoint(self, model_dir: str) -> None:
        model_path = os.path.join(model_dir, self._DARTS_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Darts checkpoint file not found: {model_path}. "
                f"Expected filename: {self._DARTS_MODEL_FILENAME}"
            )
        self.model = self._load_darts_model(model_path)
        self.is_fitted = True
        logger.info("Darts model checkpoint loaded from %s", model_path)
