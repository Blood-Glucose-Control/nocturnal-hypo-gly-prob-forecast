"""Tests for DILATE shape-aware forecast evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics.shape import (
    compute_dilate_metrics,
    compute_dilate_metrics_batch,
    DILATE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Case 1: identical sequences → all losses ≈ 0
IDENTICAL_SEQ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Case 2: constant +1 shift → shape ≈ 2.0, temporal ≈ 0.16, dilate ≈ 1.08
SHIFTED_PRED = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
SHIFTED_ACTUAL = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class TestComputeDilateMetrics:
    """Tests for the single-episode compute_dilate_metrics function."""

    def test_identical_sequences_near_zero(self):
        """When pred == actual, all DILATE components should be ≈ 0."""
        result = compute_dilate_metrics(IDENTICAL_SEQ, IDENTICAL_SEQ)

        assert set(result.keys()) == set(DILATE_COLUMNS)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite"
            assert abs(val) < 1e-3, f"{key} = {val}, expected ≈ 0"

    def test_shifted_sequence_known_values(self):
        """A constant +1 shift gives known shape, temporal, and dilate values.

        With alpha=0.5, gamma=0.001 the alignment is essentially hard DTW:
          shape ≈ 2.0  (sum of squared distances along the optimal path)
          temporal ≈ 0.16  (TDI reflecting 1-step misalignment)
          dilate = 0.5 * 2.0 + 0.5 * 0.16 = 1.08
        """
        result = compute_dilate_metrics(SHIFTED_PRED, SHIFTED_ACTUAL)

        assert set(result.keys()) == set(DILATE_COLUMNS)

        # Check hard-DTW regime (gamma=0.001); tolerance loosened to 1e-4 to
        # guard against minor float differences across numba versions/platforms.
        assert result["shape_g0001"] == pytest.approx(2.0, abs=1e-4)
        assert result["temporal_g0001"] == pytest.approx(0.16, abs=1e-4)
        # Assert the DILATE composition invariant rather than a hardcoded value.
        alpha = 0.5
        assert result["dilate_g0001"] == pytest.approx(
            alpha * result["shape_g0001"] + (1 - alpha) * result["temporal_g0001"],
            abs=1e-12,
        )

    def test_too_short_input_returns_nan(self):
        """Inputs shorter than 2 elements should return all NaN."""
        result = compute_dilate_metrics(np.array([1.0]), np.array([2.0]))
        assert all(np.isnan(v) for v in result.values())

    def test_mismatched_lengths_raises(self):
        """pred and actual with different lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_dilate_metrics(IDENTICAL_SEQ, IDENTICAL_SEQ[:3])

    def test_scalar_input_returns_nan(self):
        """Scalar (0-D) inputs are ravelled to length-1 arrays and return all NaN."""
        result = compute_dilate_metrics(np.float64(1.0), np.float64(2.0))
        assert all(np.isnan(v) for v in result.values())


class TestComputeDilateMetricsBatch:
    """Tests for the batch compute_dilate_metrics_batch function."""

    def test_batch_matches_single(self):
        """Batch results must exactly match per-episode single calls."""
        single_identical = compute_dilate_metrics(IDENTICAL_SEQ, IDENTICAL_SEQ)
        single_shifted = compute_dilate_metrics(SHIFTED_PRED, SHIFTED_ACTUAL)

        preds = np.stack([IDENTICAL_SEQ, SHIFTED_PRED])
        actuals = np.stack([IDENTICAL_SEQ, SHIFTED_ACTUAL])
        batch = compute_dilate_metrics_batch(preds, actuals)

        assert set(batch.keys()) == set(DILATE_COLUMNS)

        for key in DILATE_COLUMNS:
            assert batch[key][0] == pytest.approx(
                single_identical[key], abs=1e-12
            ), f"batch[0] mismatch on {key}"
            assert batch[key][1] == pytest.approx(
                single_shifted[key], abs=1e-12
            ), f"batch[1] mismatch on {key}"

    def test_batch_shape(self):
        """Each value array in the batch result has shape (B,)."""
        B = 3
        preds = np.tile(IDENTICAL_SEQ, (B, 1))
        actuals = np.tile(IDENTICAL_SEQ, (B, 1))
        batch = compute_dilate_metrics_batch(preds, actuals)

        for key, arr in batch.items():
            assert arr.shape == (B,), f"{key} shape = {arr.shape}, expected ({B},)"

    def test_1d_preds_raises(self):
        """1-D preds (missing batch dimension) must raise ValueError."""
        with pytest.raises(ValueError, match="preds must be 2-D"):
            compute_dilate_metrics_batch(IDENTICAL_SEQ, IDENTICAL_SEQ[np.newaxis, :])

    def test_1d_actuals_raises(self):
        """1-D actuals (missing batch dimension) must raise ValueError."""
        with pytest.raises(ValueError, match="actuals must be 2-D"):
            compute_dilate_metrics_batch(IDENTICAL_SEQ[np.newaxis, :], IDENTICAL_SEQ)

    def test_mismatched_shapes_raises(self):
        """Mismatched forecast lengths between preds and actuals must raise ValueError."""
        preds = np.tile(IDENTICAL_SEQ, (2, 1))  # (2, 5)
        actuals = np.tile(IDENTICAL_SEQ[:3], (2, 1))  # (2, 3)
        with pytest.raises(ValueError, match="identical shapes"):
            compute_dilate_metrics_batch(preds, actuals)
