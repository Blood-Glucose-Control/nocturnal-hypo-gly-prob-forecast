"""Runtime smoke tests for TimesFM 3.0 zero-shot inference and LoRA fine-tuning.

These tests require the ``timesfm3`` + ``peft`` runtime and a downloadable
pretrained checkpoint, so they are skipped (via ``importorskip``) in the shared
``.noctprob-venv``. Run them in ``.venvs/timesfm3`` (see ``make test-timesfm3``).

The covariate path dispatches through a Triton ``bmm`` override; set
``TORCH_DISABLE_NATIVE_JIT=1`` before importing torch on hosts without CUDA/Python
build headers (the Makefile target and this module both set it defensively).
"""

import os

# Must be set before torch is imported to force the covariate `bmm` path onto
# standard aten kernels on hosts lacking Triton build headers.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("timesfm3")
pytest.importorskip("peft")

from src.models.timesfm3 import TimesFM3Config, TimesFM3Forecaster  # noqa: E402

# Small context/horizon keep the smoke fast while exercising the real model.
CTX = 128
H = 24


def _make_config(**overrides) -> TimesFM3Config:
    params = dict(
        context_length=CTX,
        forecast_length=H,
        covariate_cols=["iob"],
        num_epochs=2,
        batch_size=8,
        val_fraction=0.25,
        early_stopping_patience=0,  # run both epochs for the smoke
    )
    params.update(overrides)
    return TimesFM3Config(**params)


def _synthetic_frame(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Build a CGM-like series (bg_mM) with a past-only IOB covariate."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    bg = 7.0 + 2.0 * np.sin(t / 12.0) + rng.standard_normal(n_rows) * 0.3
    iob = np.clip(rng.standard_normal(n_rows).cumsum() * 0.05 + 1.0, 0.0, None)
    return pd.DataFrame({"bg_mM": bg.astype(float), "iob": iob.astype(float)})


def test_timesfm3_zero_shot_predict_smoke() -> None:
    """Zero-shot predict yields finite point and quantile forecasts of right shape."""
    model = TimesFM3Forecaster(_make_config())
    context = _synthetic_frame(CTX)

    point = model.predict(context)
    assert point.shape == (H,)
    assert np.isfinite(point).all()

    quantiles = model.predict(context, quantile_levels=[0.1, 0.5, 0.9])
    assert quantiles.shape == (3, H)
    assert np.isfinite(quantiles).all()
    # Quantiles are monotonically non-decreasing across levels at each step.
    assert np.all(np.diff(quantiles, axis=0) >= -1e-4)


def test_timesfm3_finetune_smoke(tmp_path) -> None:
    """A 2-epoch LoRA fine-tune produces finite loss, trains adapters, and saves."""
    model = TimesFM3Forecaster(_make_config())

    # ~16 windows: context + horizon + 15 * stride(=horizon) rows.
    train_df = _synthetic_frame(CTX + H + 15 * H, seed=1)
    metrics = model.fit(train_df, output_dir=str(tmp_path))

    history = metrics["training_history"]
    assert len(history) == 2  # both epochs ran (early stopping disabled)
    for entry in history:
        assert np.isfinite(entry["train_loss"])
    assert np.isfinite(metrics["train_metrics"]["final_train_loss"])

    # LoRA adapters were injected and marked trainable.
    assert model._lora_injected is True
    trainable = [
        name
        for name, param in model.model.named_parameters()
        if "lora_" in name and param.requires_grad
    ]
    assert len(trainable) > 0

    # Only the adapter weights are persisted by _save_checkpoint.
    assert (tmp_path / "lora_adapter.pt").is_file()
    assert model.is_fitted is True

    # The fine-tuned model still produces a valid forecast.
    forecast = model.predict(_synthetic_frame(CTX, seed=2))
    assert forecast.shape == (H,)
    assert np.isfinite(forecast).all()
