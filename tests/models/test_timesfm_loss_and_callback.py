"""Unit tests for TimesFM loss functions and MidTrainingEvalCallback.

Tests cover:
  - All six loss_fn variants produce a finite scalar and support .backward()
  - MidTrainingEvalCallback writes the correct CSV header on init and appends
    a correctly formatted row on on_epoch_end
"""

import importlib.util
import os
import tempfile
import types
import unittest.mock as mock

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires torch",
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BATCH_SIZE = 4
_CONTEXT_LEN = 64
_HORIZON = 16
_MODEL_HORIZON = 128  # TimesFM native output length
_N_Q = 9  # standard TimesFM quantile heads (0.1..0.9)
_Q_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_model_mock():
    """Minimal mock for hf_prediction_model that returns stub tensors."""
    import torch

    hf = mock.MagicMock()
    hf.config.quantiles = _Q_LEVELS

    def _side_effect(past_values=None, freq=None, **kwargs):
        b = len(past_values) if past_values is not None else _BATCH_SIZE
        mean_preds = torch.randn(b, _MODEL_HORIZON, requires_grad=True)
        full_preds = torch.randn(b, _MODEL_HORIZON, _N_Q + 1, requires_grad=True)
        return types.SimpleNamespace(
            mean_predictions=mean_preds,
            full_predictions=full_preds,
        )

    hf.side_effect = _side_effect
    return hf


@pytest.fixture
def make_trainer_model(hf_model_mock):
    """Factory: returns a TimesFMForTrainer initialised with the given loss_fn."""

    def _factory(loss_fn: str, **kwargs):
        from src.models.timesfm.model import TimesFMForTrainer

        return TimesFMForTrainer(hf_model_mock, loss_fn=loss_fn, **kwargs)

    return _factory


def _make_batch(
    batch_size: int = _BATCH_SIZE, context: int = _CONTEXT_LEN, horizon: int = _HORIZON
):
    """Return (past_values list, padding, freq, future_values) stub tensors."""
    import torch

    past_values = [torch.randn(context) for _ in range(batch_size)]
    past_values_padding = torch.zeros(batch_size, context, dtype=torch.long)
    freq = torch.zeros(batch_size, dtype=torch.long)
    future_values = torch.randn(batch_size, horizon)
    return past_values, past_values_padding, freq, future_values


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss_fn",
    ["pinball", "mse", "joint", "dilate", "dilate_pinball", "dilate_pinball_median"],
)
def test_loss_fn_finite_scalar_and_backward(make_trainer_model, loss_fn):
    """Each loss_fn variant must produce a finite scalar loss and support backward."""
    import torch

    model = make_trainer_model(loss_fn=loss_fn)
    past, padding, freq, future = _make_batch()

    result = model(past, padding, freq, future_values=future)

    loss = result["loss"]
    assert loss is not None, f"loss_fn={loss_fn!r}: forward returned no loss"
    assert (
        loss.ndim == 0
    ), f"loss_fn={loss_fn!r}: expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(
        loss
    ).item(), f"loss_fn={loss_fn!r}: loss is not finite ({loss.item():.6f})"
    loss.backward()


def test_loss_fn_no_future_values_returns_no_loss(make_trainer_model):
    """When future_values is not provided, loss should be None."""
    model = make_trainer_model(loss_fn="pinball")
    past, padding, freq, _ = _make_batch()

    result = model(past, padding, freq, future_values=None)
    assert result["loss"] is None


# ---------------------------------------------------------------------------
# MidTrainingEvalCallback tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires transformers",
)
class TestMidTrainingEvalCallback:
    def _make_callback(self, tmp_dir: str):
        from src.models.timesfm.model import MidTrainingEvalCallback

        return MidTrainingEvalCallback(
            eval_dataset=None,
            output_dir=tmp_dir,
            horizon=_HORIZON,
            quantile_levels=_Q_LEVELS,
            collate_fn=lambda x: x,
            batch_size=4,
            device="cpu",
        )

    def test_init_writes_csv_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._make_callback(tmp_dir)
            csv_path = os.path.join(tmp_dir, "epoch_metrics.csv")
            assert os.path.exists(csv_path), "epoch_metrics.csv not created on init"
            with open(csv_path) as f:
                header = f.readline().strip()
            assert (
                header
                == "epoch,train_loss,wql,coverage_50,coverage_80,coverage_95,mace,rmse"
            )

    def test_init_does_not_overwrite_existing_header(self):
        """Re-initialising the callback (e.g. after checkpoint resume) must not
        overwrite an existing CSV with data in it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "epoch_metrics.csv")
            with open(csv_path, "w") as f:
                f.write(
                    "epoch,train_loss,wql,coverage_50,coverage_80,coverage_95,mace,rmse\n"
                )
                f.write("1,0.500000,0.300000,0.4800,0.7900,0.9400,0.200000,1.200000\n")

            self._make_callback(tmp_dir)  # second init — should not truncate

            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 2, "Existing CSV data was overwritten on re-init"

    def test_on_epoch_end_appends_valid_row(self):
        """on_epoch_end should append a row with 8 comma-separated numeric fields."""
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            from src.models.timesfm.model import MidTrainingEvalCallback

            # Build a tiny eval dataset with 2 windows
            horizon = 4
            n_q = len(_Q_LEVELS)
            windows = []
            for _ in range(2):
                context = np.random.randn(_CONTEXT_LEN).astype(np.float32)
                future = np.random.randn(horizon).astype(np.float32)
                windows.append((context, future))

            class _TinyDataset:
                def __len__(self):
                    return len(windows)

                def __getitem__(self, idx):
                    ctx, fut = windows[idx]
                    return {
                        "past_values": torch.tensor(ctx),
                        "past_values_padding": torch.zeros(len(ctx), dtype=torch.long),
                        "freq": torch.tensor(0, dtype=torch.long),
                        "future_values": torch.tensor(fut),
                    }

            # Mock a trainer model with stub prediction outputs
            model_mock = mock.MagicMock()
            model_mock.eval = mock.MagicMock()
            model_mock.train = mock.MagicMock()

            def _pred_side_effect(past_values=None, freq=None, **kw):
                b = len(past_values)
                mean_preds = torch.randn(b, _MODEL_HORIZON)
                full_preds = torch.randn(b, _MODEL_HORIZON, n_q + 1)
                return types.SimpleNamespace(
                    mean_predictions=mean_preds,
                    full_predictions=full_preds,
                )

            model_mock.prediction_model.side_effect = _pred_side_effect

            # Collate fn: same as TimesFMForecaster._collate_fn
            def _collate(batch):
                return {
                    "past_values": [item["past_values"] for item in batch],
                    "past_values_padding": torch.stack(
                        [item["past_values_padding"] for item in batch]
                    ),
                    "freq": torch.stack([item["freq"] for item in batch]),
                    "future_values": torch.stack(
                        [item["future_values"] for item in batch]
                    ),
                }

            cb = MidTrainingEvalCallback(
                eval_dataset=_TinyDataset(),
                output_dir=tmp_dir,
                horizon=horizon,
                quantile_levels=_Q_LEVELS,
                collate_fn=_collate,
                batch_size=2,
                device="cpu",
            )

            # Build minimal state / args mocks
            args_mock = mock.MagicMock()
            args_mock.bf16 = False
            args_mock.fp16 = False

            state_mock = mock.MagicMock()
            state_mock.epoch = 1.0
            state_mock.log_history = [{"loss": 0.42}]

            cb.on_epoch_end(args_mock, state_mock, None, model=model_mock)

            csv_path = os.path.join(tmp_dir, "epoch_metrics.csv")
            with open(csv_path) as f:
                lines = f.readlines()

            # Header + 1 data row
            assert len(lines) == 2, f"Expected 2 lines in CSV, got {len(lines)}"
            row = lines[1].strip().split(",")
            assert (
                len(row) == 8
            ), f"Expected 8 columns in CSV row, got {len(row)}: {row}"
            # All fields should be parseable as numbers (epoch is int-like, rest are floats)
            for val in row:
                float(val)  # raises if not numeric
