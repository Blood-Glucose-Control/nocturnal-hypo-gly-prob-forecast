"""Tests for probabilistic forecast evaluation metrics (WQL and Brier score)."""

import numpy as np
import pytest

from src.evaluation.metrics.probabilistic import (
    compute_wql,
    compute_brier_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _perfect_quantile_forecast(actuals: np.ndarray) -> np.ndarray:
    """Build a 'perfect' quantile forecast where every quantile equals actuals."""
    # shape: (n_quantiles, forecast_length)
    return np.tile(actuals, (len(QUANTILE_LEVELS), 1))


# ---------------------------------------------------------------------------
# compute_wql tests
# ---------------------------------------------------------------------------


class TestComputeWQL:
    def test_perfect_forecast_is_zero(self):
        actuals = np.array([5.0, 6.0, 7.0, 4.0])
        q_forecast = _perfect_quantile_forecast(actuals)
        wql = compute_wql(q_forecast, actuals, QUANTILE_LEVELS)
        assert wql == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_imperfect_forecast(self):
        actuals = np.array([5.0, 6.0, 7.0])
        q_forecast = np.full((len(QUANTILE_LEVELS), 3), 5.0)  # constant forecast
        wql = compute_wql(q_forecast, actuals, QUANTILE_LEVELS)
        assert wql > 0.0

    def test_symmetric_overunder(self):
        """Pinball loss at q=0.5 is symmetric: over-predict by 1 == under-predict by 1."""
        actuals = np.array([5.0])
        over = np.array([[6.0]])  # single quantile, single step
        under = np.array([[4.0]])
        wql_over = compute_wql(over, actuals, [0.5])
        wql_under = compute_wql(under, actuals, [0.5])
        assert wql_over == pytest.approx(wql_under)

    def test_low_quantile_penalises_overpredict_more(self):
        """At q=0.1, over-predicting should cost more than under-predicting."""
        actuals = np.array([5.0])
        over = np.array([[6.0]])
        under = np.array([[4.0]])
        wql_over = compute_wql(over, actuals, [0.1])
        wql_under = compute_wql(under, actuals, [0.1])
        assert wql_over > wql_under

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2D"):
            compute_wql(np.array([1, 2, 3]), np.array([1, 2, 3]), [0.5])
        with pytest.raises(ValueError, match="does not match"):
            compute_wql(np.ones((3, 4)), np.ones(4), [0.1, 0.5])  # 3 != 2 quantiles

    def test_known_value(self):
        """Hand-computed pinball loss for a simple case."""
        # q=0.5, forecast=4.0, actual=5.0 → pinball = 0.5 * (5-4) = 0.5
        # q=0.5, forecast=4.0, actual=3.0 → pinball = 0.5 * (4-3) = 0.5
        q_forecast = np.array([[4.0, 4.0]])
        actuals = np.array([5.0, 3.0])
        wql = compute_wql(q_forecast, actuals, [0.5])
        assert wql == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_brier_score tests
# ---------------------------------------------------------------------------


class TestComputeBrierScore:
    def test_perfect_calibration_no_hypo(self):
        """All actuals well above threshold, forecast shows no hypo risk → Brier ≈ 0."""
        actuals = np.array([8.0, 9.0, 10.0, 7.0])
        # All quantiles well above threshold
        q_forecast = np.linspace(6.0, 12.0, len(QUANTILE_LEVELS))[:, np.newaxis]
        q_forecast = np.tile(q_forecast, (1, len(actuals)))
        brier = compute_brier_score(q_forecast, actuals, QUANTILE_LEVELS)
        # P(hypo) should be clamped to q_min=0.1 since threshold < all quantile values
        # indicator = 0 for all → Brier = mean(0.1^2) = 0.01
        assert brier == pytest.approx(0.01, abs=0.001)

    def test_all_hypo_with_high_forecast(self):
        """All actuals below threshold but forecast predicts high → high Brier."""
        actuals = np.array([2.0, 3.0, 3.5])  # all below 3.9
        # Quantile forecast: all values well above threshold
        q_forecast = np.linspace(5.0, 10.0, len(QUANTILE_LEVELS))[:, np.newaxis]
        q_forecast = np.tile(q_forecast, (1, len(actuals)))
        brier = compute_brier_score(q_forecast, actuals, QUANTILE_LEVELS)
        # P(hypo) clamped to 0.1, but indicator=1 → Brier = mean((0.1-1)^2) = 0.81
        assert brier == pytest.approx(0.81, abs=0.01)

    def test_threshold_default(self):
        """Default threshold should be 3.9 mmol/L."""
        actuals = np.array([5.0])
        q_forecast = np.linspace(4.0, 8.0, len(QUANTILE_LEVELS)).reshape(-1, 1)
        brier_default = compute_brier_score(q_forecast, actuals, QUANTILE_LEVELS)
        brier_explicit = compute_brier_score(
            q_forecast, actuals, QUANTILE_LEVELS, threshold=3.9
        )
        assert brier_default == pytest.approx(brier_explicit)

    def test_brier_bounded_0_1(self):
        """Brier score should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            actuals = rng.uniform(2.0, 15.0, size=72)
            q_vals = np.sort(
                rng.uniform(2.0, 15.0, size=(len(QUANTILE_LEVELS), 72)), axis=0
            )
            brier = compute_brier_score(q_vals, actuals, QUANTILE_LEVELS)
            assert 0.0 <= brier <= 1.0

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2D"):
            compute_brier_score(np.array([1, 2, 3]), np.array([1, 2, 3]), [0.5])

    def test_unsorted_quantile_levels_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_brier_score(
                np.ones((2, 3)),
                np.ones(3),
                [0.9, 0.1],  # descending
            )
