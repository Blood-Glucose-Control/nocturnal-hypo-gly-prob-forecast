"""Tests for probabilistic forecast evaluation metrics (WQL and Brier score)."""

import numpy as np
import pytest

from src.evaluation.metrics.probabilistic import (
    compute_wql,
    compute_brier_score,
    compute_coverage,
    compute_sharpness,
    compute_mace,
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


# ---------------------------------------------------------------------------
# compute_coverage tests
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_perfect_coverage(self):
        """Actual always inside interval → coverage = 1.0."""
        actuals = np.array([5.0, 5.0, 5.0])
        # Quantile forecasts: wide intervals centered on 5.0
        q_forecast = np.array(
            [
                [1.0, 1.0, 1.0],  # q=0.1
                [3.0, 3.0, 3.0],  # q=0.25
                [5.0, 5.0, 5.0],  # q=0.5
                [7.0, 7.0, 7.0],  # q=0.75
                [9.0, 9.0, 9.0],  # q=0.9
            ]
        )
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        assert compute_coverage(q_forecast, actuals, levels, 0.5) == 1.0
        assert compute_coverage(q_forecast, actuals, levels, 0.8) == 1.0

    def test_zero_coverage_when_outside(self):
        """Actual always outside interval → coverage = 0.0."""
        actuals = np.array([100.0, 100.0, 100.0])
        q_forecast = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0],
                [5.0, 5.0, 5.0],
                [7.0, 7.0, 7.0],
                [9.0, 9.0, 9.0],
            ]
        )
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        assert compute_coverage(q_forecast, actuals, levels, 0.5) == 0.0

    def test_interpolates_when_target_missing(self):
        """When target quantiles (e.g., 0.25 for level=0.5) are absent,
        bounds should be linearly interpolated — no silent asymmetry."""
        # Available levels: [0.1, 0.2, ..., 0.9] — 0.25 and 0.75 absent
        levels = [round(0.1 * i, 1) for i in range(1, 10)]
        # Quantile forecasts: q_k = k (so q=0.1 → 1, q=0.2 → 2, ..., q=0.9 → 9)
        q_forecast = np.array([[float(q * 10)] for q in levels])
        # For level=0.5 we want bounds at q=0.25 and q=0.75
        # Interpolated: q=0.25 → 2.5, q=0.75 → 7.5
        actuals_in = np.array([5.0])  # inside [2.5, 7.5]
        actuals_out = np.array([2.0])  # outside (below 2.5)
        assert compute_coverage(q_forecast, actuals_in, levels, 0.5) == 1.0
        assert compute_coverage(q_forecast, actuals_out, levels, 0.5) == 0.0

    def test_invalid_level_raises(self):
        q_forecast = np.ones((3, 2))
        actuals = np.ones(2)
        with pytest.raises(ValueError, match="level must be in"):
            compute_coverage(q_forecast, actuals, [0.1, 0.5, 0.9], level=0.0)
        with pytest.raises(ValueError, match="level must be in"):
            compute_coverage(q_forecast, actuals, [0.1, 0.5, 0.9], level=1.0)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2D"):
            compute_coverage(np.array([1, 2, 3]), np.array([1, 2, 3]), [0.5])
        with pytest.raises(ValueError, match="does not match"):
            compute_coverage(np.ones((3, 4)), np.ones(4), [0.1, 0.5])

    def test_unsorted_quantile_levels_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_coverage(np.ones((2, 3)), np.ones(3), [0.9, 0.1])


# ---------------------------------------------------------------------------
# compute_sharpness tests
# ---------------------------------------------------------------------------


class TestComputeSharpness:
    def test_collapsed_forecast_has_zero_sharpness(self):
        """All quantiles equal → interval width = 0."""
        q_forecast = np.full((5, 3), 5.0)
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        assert compute_sharpness(q_forecast, levels, 0.5) == 0.0
        assert compute_sharpness(q_forecast, levels, 0.8) == 0.0

    def test_known_width(self):
        """Quantile gap of 4 at level=0.5 (q=0.25/0.75) → sharpness = 4."""
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        q_forecast = np.array(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 5.0],
                [7.0, 7.0],
                [9.0, 9.0],
            ]
        )
        assert compute_sharpness(q_forecast, levels, 0.5) == pytest.approx(4.0)
        # level=0.8 → q=0.1/0.9 → width = 9-1 = 8
        assert compute_sharpness(q_forecast, levels, 0.8) == pytest.approx(8.0)

    def test_wider_level_is_wider(self):
        """Sharpness at 80% should be >= sharpness at 50%."""
        rng = np.random.default_rng(0)
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        q_forecast = np.sort(rng.uniform(0, 10, size=(5, 20)), axis=0)
        s50 = compute_sharpness(q_forecast, levels, 0.5)
        s80 = compute_sharpness(q_forecast, levels, 0.8)
        assert s80 >= s50

    def test_invalid_level_raises(self):
        q_forecast = np.ones((3, 2))
        with pytest.raises(ValueError, match="level must be in"):
            compute_sharpness(q_forecast, [0.1, 0.5, 0.9], level=1.5)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2D"):
            compute_sharpness(np.array([1, 2, 3]), [0.5])


# ---------------------------------------------------------------------------
# compute_mace tests
# ---------------------------------------------------------------------------


class TestComputeMACE:
    def test_perfect_calibration(self):
        """If forecast_q equals empirical q-th quantile of actuals → MACE = 0."""
        # Actuals drawn from uniform → empirical q-th quantile ≈ q * (max - min) + min
        rng = np.random.default_rng(0)
        n = 10000
        actuals = rng.uniform(0, 10, size=n)
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Set each forecast quantile to the empirical quantile of actuals
        q_forecast = np.array([[np.quantile(actuals, q)] * n for q in levels])
        mace = compute_mace(q_forecast, actuals, levels)
        assert mace < 0.01  # approximately perfectly calibrated

    def test_systematic_bias_gives_positive_mace(self):
        """Forecast quantiles constant and wrong → MACE > 0."""
        actuals = np.array([5.0] * 100)
        # All quantiles predict 10.0 → empirical P(actual <= 10) = 1.0 for every q
        # MACE = mean(|1 - q| for q in levels)
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        q_forecast = np.full((5, 100), 10.0)
        mace = compute_mace(q_forecast, actuals, levels)
        expected = np.mean([abs(1.0 - q) for q in levels])
        assert mace == pytest.approx(expected)

    def test_mace_bounded_0_1(self):
        """MACE should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for _ in range(20):
            actuals = rng.uniform(2.0, 15.0, size=72)
            q_vals = np.sort(rng.uniform(2.0, 15.0, size=(len(levels), 72)), axis=0)
            mace = compute_mace(q_vals, actuals, levels)
            assert 0.0 <= mace <= 1.0

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2D"):
            compute_mace(np.array([1, 2, 3]), np.array([1, 2, 3]), [0.5])
        with pytest.raises(ValueError, match="does not match"):
            compute_mace(np.ones((3, 4)), np.ones(4), [0.1, 0.5])

    def test_unsorted_quantile_levels_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_mace(np.ones((2, 3)), np.ones(3), [0.9, 0.1])
