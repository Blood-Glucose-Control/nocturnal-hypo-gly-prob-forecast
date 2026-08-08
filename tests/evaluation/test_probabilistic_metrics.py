"""Tests for probabilistic forecast evaluation metrics (WQL and Brier score)."""

import numpy as np
import pytest

from src.evaluation.metrics.probabilistic import (
    compute_wql,
    compute_brier_score,
    compute_coverage,
    compute_coverage_by_step,
    compute_sharpness,
    compute_sharpness_by_step,
    compute_mace,
    compute_pit_values,
    compute_reliability_curve,
    compute_ece,
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

    def test_crossing_quantiles_handled_correctly(self):
        """Crossing quantile BG values must be handled correctly by sorting x_vals.

        The fix sorts only the BG values (x_vals) while keeping q_arr in its
        original increasing order, producing a valid monotone CDF.  Sorting
        both arrays together (q_arr[sort_idx]) was wrong because it made the
        implied CDF non-monotone (probabilities could decrease as BG increased).
        """
        actuals = np.array([4.5])  # above threshold 3.9 → indicator = 0
        levels = [0.1, 0.5, 0.9]
        # q50=4.1 > q90=4.0 — a typical mild TimesFM crossing
        q_crossed = np.array([[3.0], [4.1], [4.0]])
        # Expected: np.sort([3.0, 4.1, 4.0]) = [3.0, 4.0, 4.1] paired with q_arr [0.1, 0.5, 0.9]
        x_sorted = np.array([3.0, 4.0, 4.1])
        q_arr = np.array([0.1, 0.5, 0.9])  # unchanged — always in original order
        expected_p = float(np.interp(3.9, x_sorted, q_arr, left=0.1, right=0.9))
        expected_brier = expected_p**2  # (P_hat - 0)^2
        brier = compute_brier_score(q_crossed, actuals, levels)
        assert brier == pytest.approx(expected_brier, rel=1e-9)


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
# per-step coverage / sharpness tests
# ---------------------------------------------------------------------------


class TestPerStepProbabilisticMetrics:
    @staticmethod
    def _coverage_by_step_reference(
        q_batch: np.ndarray,
        actuals: np.ndarray,
        q_levels: list,
        level: float,
    ) -> np.ndarray:
        """Slow reference implementation used for regression parity checks."""
        q_arr = np.array(q_levels, dtype=float)
        target_lower = (1.0 - level) / 2.0
        target_upper = (1.0 + level) / 2.0
        n_eps, _, fh = q_batch.shape

        out = np.empty(fh, dtype=float)
        for t in range(fh):
            covered = 0
            for e in range(n_eps):
                lo = np.interp(target_lower, q_arr, q_batch[e, :, t])
                hi = np.interp(target_upper, q_arr, q_batch[e, :, t])
                covered += int(lo <= actuals[e, t] <= hi)
            out[t] = covered / n_eps
        return out

    @staticmethod
    def _sharpness_by_step_reference(
        q_batch: np.ndarray,
        q_levels: list,
        level: float,
    ) -> np.ndarray:
        """Slow reference implementation used for regression parity checks."""
        q_arr = np.array(q_levels, dtype=float)
        target_lower = (1.0 - level) / 2.0
        target_upper = (1.0 + level) / 2.0
        n_eps, _, fh = q_batch.shape

        out = np.empty(fh, dtype=float)
        for t in range(fh):
            widths = []
            for e in range(n_eps):
                lo = np.interp(target_lower, q_arr, q_batch[e, :, t])
                hi = np.interp(target_upper, q_arr, q_batch[e, :, t])
                widths.append(hi - lo)
            out[t] = float(np.mean(widths))
        return out

    def test_coverage_by_step_known_values(self):
        """Per-step coverage matches hand-computed fractions across episodes."""
        # 2 episodes, 3 quantiles [0.1, 0.5, 0.9], 4-step horizon
        # level=0.8 -> bounds are q0.1 and q0.9 exactly
        q_levels = [0.1, 0.5, 0.9]
        q_batch = np.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
                [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            ],
            dtype=float,
        )
        # Expected covered fractions by step: [1.0, 0.5, 1.0, 0.0]
        actuals = np.array(
            [
                [2.5, 5.0, 4.2, 10.0],
                [1.5, 2.5, 3.5, 10.0],
            ],
            dtype=float,
        )

        coverage = compute_coverage_by_step(q_batch, actuals, q_levels, level=0.8)
        assert np.allclose(coverage, np.array([1.0, 0.5, 1.0, 0.0]))

    def test_sharpness_by_step_known_values(self):
        """Per-step sharpness equals mean interval width at each horizon step."""
        q_levels = [0.1, 0.5, 0.9]
        q_batch = np.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
                [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            ],
            dtype=float,
        )
        # Widths are always 2.0 for both episodes at all steps -> per-step mean is 2.0
        sharpness = compute_sharpness_by_step(q_batch, q_levels, level=0.8)
        assert np.allclose(sharpness, np.array([2.0, 2.0, 2.0, 2.0]))

    def test_per_step_level_validation(self):
        q_levels = [0.1, 0.5, 0.9]
        q_batch = np.ones((2, 3, 4))
        actuals = np.ones((2, 4))

        with pytest.raises(ValueError, match="level must be in"):
            compute_coverage_by_step(q_batch, actuals, q_levels, level=0.0)
        with pytest.raises(ValueError, match="level must be in"):
            compute_sharpness_by_step(q_batch, q_levels, level=1.0)

    def test_per_step_quantile_level_validation(self):
        q_batch = np.ones((2, 2, 4))
        actuals = np.ones((2, 4))

        with pytest.raises(ValueError, match="strictly increasing"):
            compute_coverage_by_step(q_batch, actuals, [0.9, 0.1], level=0.8)
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_sharpness_by_step(q_batch, [0.9, 0.1], level=0.8)

    def test_per_step_shape_validation(self):
        q_levels = [0.1, 0.5, 0.9]

        with pytest.raises(ValueError, match="must be 3D"):
            compute_coverage_by_step(
                np.ones((2, 3)), np.ones((2, 4)), q_levels, level=0.8
            )
        with pytest.raises(ValueError, match="must be 3D"):
            compute_sharpness_by_step(np.ones((2, 3)), q_levels, level=0.8)

    def test_per_step_actuals_shape_validation(self):
        q_levels = [0.1, 0.5, 0.9]
        q_batch = np.ones((2, 3, 4))

        with pytest.raises(ValueError, match="actuals_batch must be 2D"):
            compute_coverage_by_step(q_batch, np.ones(4), q_levels, level=0.8)
        with pytest.raises(ValueError, match=r"shape\[0\].*does not match"):
            compute_coverage_by_step(q_batch, np.ones((3, 4)), q_levels, level=0.8)
        with pytest.raises(ValueError, match=r"shape\[1\].*does not match"):
            compute_coverage_by_step(q_batch, np.ones((2, 5)), q_levels, level=0.8)

    def test_per_step_quantile_count_validation(self):
        q_batch = np.ones((2, 3, 4))
        actuals = np.ones((2, 4))

        with pytest.raises(ValueError, match=r"len\(quantile_levels\).+does not match"):
            compute_coverage_by_step(q_batch, actuals, [0.1, 0.5], level=0.8)
        with pytest.raises(ValueError, match=r"len\(quantile_levels\).+does not match"):
            compute_sharpness_by_step(q_batch, [0.1, 0.5], level=0.8)

    def test_per_step_randomized_parity_with_reference(self):
        """Vectorized/optimized implementations must match reference outputs."""
        rng = np.random.default_rng(123)
        n_eps, n_q, fh = 11, 9, 17
        q_levels = [round(0.1 * i, 1) for i in range(1, 10)]
        q_batch = np.sort(rng.uniform(2.0, 14.0, size=(n_eps, n_q, fh)), axis=1)
        actuals = rng.uniform(1.5, 14.5, size=(n_eps, fh))
        level = 0.65  # Forces interpolation (targets 0.175 / 0.825 are absent)

        cov_expected = self._coverage_by_step_reference(
            q_batch, actuals, q_levels, level
        )
        shp_expected = self._sharpness_by_step_reference(q_batch, q_levels, level)

        cov_got = compute_coverage_by_step(q_batch, actuals, q_levels, level=level)
        shp_got = compute_sharpness_by_step(q_batch, q_levels, level=level)

        assert np.allclose(cov_got, cov_expected, atol=1e-12)
        assert np.allclose(shp_got, shp_expected, atol=1e-12)

    def test_per_step_missing_target_quantiles_interpolate_correctly(self):
        """Interpolation works when interval bounds are not explicit quantile levels."""
        # level=0.5 requires 0.25/0.75, which are absent here.
        q_levels = [round(0.1 * i, 1) for i in range(1, 10)]
        q_batch = np.array(
            [
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
            ],
            dtype=float,
        )
        actuals = np.array([[5.0], [2.0]], dtype=float)

        # Interpolated bounds: lower=2.5 at q=0.25, upper=7.5 at q=0.75
        coverage = compute_coverage_by_step(q_batch, actuals, q_levels, level=0.5)
        sharpness = compute_sharpness_by_step(q_batch, q_levels, level=0.5)

        assert np.allclose(coverage, np.array([0.5]))
        assert np.allclose(sharpness, np.array([5.0]))


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


# ---------------------------------------------------------------------------
# compute_pit_values tests
# ---------------------------------------------------------------------------


def _batch_quantile_forecast(
    actuals_batch: np.ndarray,
    spread: float = 1.0,
) -> np.ndarray:
    """Build a batch forecast: quantiles evenly spread around each actual.

    Returns shape (n_episodes, n_quantiles, forecast_length).
    """
    n_eps, fh = actuals_batch.shape
    n_q = len(QUANTILE_LEVELS)
    # Spread quantiles symmetrically: q_i ∈ [actual - spread, actual + spread]
    offsets = np.linspace(-spread, spread, n_q)  # (n_q,)
    # broadcast: (n_eps, n_q, fh)
    return actuals_batch[:, np.newaxis, :] + offsets[np.newaxis, :, np.newaxis]


class TestComputePITValues:
    def test_output_shape_is_flat(self):
        """Returns flat (n_episodes * forecast_length,) array."""
        n_eps, fh = 5, 12
        actuals = np.ones((n_eps, fh)) * 5.0
        q_fc = _batch_quantile_forecast(actuals)
        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        assert pit.shape == (n_eps * fh,)

    def test_uniform_spread_yields_approx_uniform(self):
        """Uniform actuals spanning the quantile range → PIT ≈ uniform in [0.1, 0.9]."""
        rng = np.random.default_rng(0)
        n_eps, fh = 1000, 96
        n_q = len(QUANTILE_LEVELS)

        # Fixed quantile forecast: q_levels[i] maps to value (i+1), so the
        # forecast CDF is a linearly-spaced grid from 1.0 to 9.0.
        q_vals = np.arange(1, n_q + 1, dtype=np.float64)  # [1, 2, ..., 9]
        q_fc = np.broadcast_to(
            q_vals[np.newaxis, :, np.newaxis], (n_eps, n_q, fh)
        ).copy()

        # Actuals drawn uniformly inside [1, 9] (the quantile value range).
        actuals = rng.uniform(1.0, 9.0, size=(n_eps, fh))

        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        # Under this setup PIT ~ Uniform[0.1, 0.9]; median ≈ 0.5.
        assert 0.4 < float(np.median(pit)) < 0.6
        assert float(np.percentile(pit, 25)) < 0.45  # ~0.3
        assert float(np.percentile(pit, 75)) > 0.55  # ~0.7

    def test_actual_at_median_quantile_gives_pit_half(self):
        """When actual == forecast for the median quantile, PIT ≈ 0.5."""
        fh = 4
        actuals = np.array([[5.0, 6.0, 7.0, 8.0]])  # (1, 4)
        # Build quantiles so q=0.5 exactly hits the actual
        n_q = len(QUANTILE_LEVELS)
        q_fc = np.zeros((1, n_q, fh))
        for i, q in enumerate(QUANTILE_LEVELS):
            q_fc[0, i, :] = actuals[0] + (q - 0.5) * 2.0
        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        assert np.allclose(pit, 0.5, atol=1e-10)

    def test_below_all_quantiles_gives_pit_zero(self):
        """Actual below the lowest quantile → PIT = 0.0."""
        actuals = np.array([[0.0]])  # (1, 1) — below all forecast values
        q_fc = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]]).reshape(
            1, len(QUANTILE_LEVELS), 1
        )
        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        assert pit[0] == pytest.approx(0.0)

    def test_above_all_quantiles_gives_pit_one(self):
        """Actual above the highest quantile → PIT = 1.0."""
        actuals = np.array([[99.0]])  # (1, 1) — above all forecast values
        q_fc = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]]).reshape(
            1, len(QUANTILE_LEVELS), 1
        )
        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        assert pit[0] == pytest.approx(1.0)

    def test_pit_bounded_0_1(self):
        """All PIT values must lie in [0, 1] for arbitrary inputs."""
        rng = np.random.default_rng(7)
        actuals = rng.uniform(2.0, 20.0, size=(20, 48))
        q_fc = np.sort(
            rng.uniform(2.0, 20.0, size=(20, len(QUANTILE_LEVELS), 48)), axis=1
        )
        pit = compute_pit_values(q_fc, actuals, QUANTILE_LEVELS)
        assert np.all(pit >= 0.0)
        assert np.all(pit <= 1.0)

    def test_linear_interpolation_between_quantiles(self):
        """PIT is linearly interpolated inside a quantile bin."""
        # Two quantiles: q=0.1 → value=1.0, q=0.2 → value=2.0
        # Actual = 1.5 → PIT should be 0.15 (midpoint of [0.1, 0.2])
        actuals = np.array([[1.5]])  # (1, 1)
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_fc = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(1, 9, 1)
        pit = compute_pit_values(q_fc, actuals, q_levels)
        assert pit[0] == pytest.approx(0.15, abs=1e-10)

    def test_shape_mismatch_raises(self):
        """Mismatched n_quantiles and len(quantile_levels) should raise ValueError."""
        q_fc = np.ones((3, 5, 10))  # 5 quantiles
        actuals = np.ones((3, 10))
        with pytest.raises(ValueError, match="does not match"):
            compute_pit_values(q_fc, actuals, [0.1, 0.5, 0.9])  # 3 levels

    def test_unsorted_quantile_levels_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            q_fc = np.ones((2, 3, 4))
            compute_pit_values(q_fc, np.ones((2, 4)), [0.9, 0.5, 0.1])

    def test_actual_at_max_quantile_yields_q_max_not_one(self):
        """y == xp.max() should yield PIT = q_arr[-1], not 1.0.

        The old code used `above = (hi == n_quantiles)` which counted equality at
        the upper boundary as "above all quantiles", producing PIT = 1.0 instead
        of the correct q_arr[-1] (e.g. 0.9).
        """
        # q_fc: single episode, single timestep, q90 = 8.0
        q_fc = np.array([[[2.0], [4.0], [6.0], [8.0]]])  # (1, 4, 1)
        q_levels = [0.1, 0.4, 0.7, 0.9]
        actuals = np.array([[8.0]])  # actual == q90 value
        pit = compute_pit_values(q_fc, actuals, q_levels)
        assert pit[0] == pytest.approx(
            0.9
        ), f"Expected PIT=0.9 (q_arr[-1]) when actual equals max quantile, got {pit[0]}"

    def test_inverted_quantile_values_raises(self):
        """Catastrophic inversions (> 1 mmol/L) should raise ValueError."""
        q_fc = np.ones((2, 3, 4))
        q_fc[:, 0, :] = 5.0  # lower quantile has higher value → inversion = 2 mmol/L
        q_fc[:, 1, :] = 3.0
        q_fc[:, 2, :] = 7.0
        with pytest.raises(ValueError, match="quantile inversions"):
            compute_pit_values(q_fc, np.ones((2, 4)), [0.1, 0.5, 0.9])

    def test_moderate_quantile_inversions_sorted(self):
        """Moderate inversions (0.01–1.0 mmol/L) should be sorted internally with a warning.

        The function now works on a copy of the input (np.array, not np.asarray),
        so the caller's original array is NOT mutated.  The function should still
        warn and complete without raising.
        """
        q_fc = np.ones((2, 3, 4))
        q_fc[:, 0, :] = 5.0  # inversion magnitude = 0.1 mmol/L (middle tier)
        q_fc[:, 1, :] = 4.9
        q_fc[:, 2, :] = 7.0
        q_fc_original = q_fc.copy()  # save to verify caller's array is not mutated
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            compute_pit_values(q_fc, np.ones((2, 4)), [0.1, 0.5, 0.9])
        assert any("sorted in-place" in str(warning.message) for warning in w)
        # Caller's original array must NOT be mutated
        np.testing.assert_array_equal(q_fc, q_fc_original)


# ---------------------------------------------------------------------------


class TestComputeReliabilityCurve:
    def test_output_shapes(self):
        """Returns two arrays each of length n_quantiles."""
        n_eps, fh = 10, 12
        actuals = np.ones((n_eps, fh)) * 5.0
        q_fc = _batch_quantile_forecast(actuals)
        nominal, empirical = compute_reliability_curve(q_fc, actuals, QUANTILE_LEVELS)
        assert nominal.shape == (len(QUANTILE_LEVELS),)
        assert empirical.shape == (len(QUANTILE_LEVELS),)

    def test_nominal_equals_quantile_levels(self):
        """nominal should be exactly the supplied quantile_levels."""
        actuals = np.ones((4, 8)) * 5.0
        q_fc = _batch_quantile_forecast(actuals)
        nominal, _ = compute_reliability_curve(q_fc, actuals, QUANTILE_LEVELS)
        np.testing.assert_array_almost_equal(nominal, QUANTILE_LEVELS)

    def test_perfect_calibration_diagonal(self):
        """When the forecast CDF exactly matches the data, empirical ≈ nominal."""
        # For a perfectly calibrated model, the predicted quantile at level q
        # should equal the true q-th quantile of the data.
        # Actuals ~ U[1, 9].  True quantile at level q = 1 + q * 8.
        rng = np.random.default_rng(42)
        n_q = len(QUANTILE_LEVELS)
        n_eps, fh = 2000, 96
        q_arr = np.array(QUANTILE_LEVELS, dtype=np.float64)  # [0.1..0.9]
        q_vals = 1.0 + q_arr * 8.0  # true quantiles of U[1,9]
        q_fc = np.broadcast_to(
            q_vals[np.newaxis, :, np.newaxis], (n_eps, n_q, fh)
        ).copy()
        actuals = rng.uniform(1.0, 9.0, size=(n_eps, fh))
        nominal, empirical = compute_reliability_curve(q_fc, actuals, QUANTILE_LEVELS)
        # P(U[1,9] ≤ 1 + q*8) = q  →  empirical ≈ nominal
        np.testing.assert_allclose(empirical, nominal, atol=0.02)

    def test_empirical_bounded_0_1(self):
        """All empirical values must lie in [0, 1]."""
        rng = np.random.default_rng(7)
        actuals = rng.uniform(2.0, 20.0, size=(20, 48))
        q_fc = np.sort(
            rng.uniform(2.0, 20.0, size=(20, len(QUANTILE_LEVELS), 48)), axis=1
        )
        _, empirical = compute_reliability_curve(q_fc, actuals, QUANTILE_LEVELS)
        assert np.all(empirical >= 0.0)
        assert np.all(empirical <= 1.0)

    def test_over_forecasting_curve_above_diagonal(self):
        """If quantile forecasts are always too high, empirical > nominal."""
        n_eps, fh = 50, 10
        actuals = np.ones((n_eps, fh)) * 3.0
        # All quantile values well above actuals → P(actual ≤ q_i) ≈ 1 for all i
        q_fc = np.ones((n_eps, len(QUANTILE_LEVELS), fh)) * 10.0
        nominal, empirical = compute_reliability_curve(q_fc, actuals, QUANTILE_LEVELS)
        assert np.all(empirical > nominal)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            q_fc = np.ones((5, 3, 10))
            compute_reliability_curve(q_fc, np.ones((5, 10)), [0.1, 0.9])

    def test_unsorted_quantile_levels_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_reliability_curve(np.ones((2, 2, 4)), np.ones((2, 4)), [0.9, 0.1])

    def test_inverted_quantile_values_raises(self):
        """Inverted quantile values should raise ValueError."""
        q_fc = np.ones((2, 2, 4))
        q_fc[:, 0, :] = 5.0  # lower quantile has higher value → inverted
        q_fc[:, 1, :] = 3.0
        with pytest.raises(ValueError, match="quantile inversions"):
            compute_reliability_curve(q_fc, np.ones((2, 4)), [0.1, 0.9])


# ---------------------------------------------------------------------------


class TestComputeECE:
    def test_perfect_calibration_ece_near_zero(self):
        """Perfect calibration → ECE ≈ 0."""
        rng = np.random.default_rng(42)
        n_q = len(QUANTILE_LEVELS)
        n_eps, fh = 2000, 96
        q_arr = np.array(QUANTILE_LEVELS, dtype=np.float64)
        q_vals = 1.0 + q_arr * 8.0  # true quantiles of U[1,9]
        q_fc = np.broadcast_to(
            q_vals[np.newaxis, :, np.newaxis], (n_eps, n_q, fh)
        ).copy()
        actuals = rng.uniform(1.0, 9.0, size=(n_eps, fh))
        ece = compute_ece(q_fc, actuals, QUANTILE_LEVELS)
        assert ece == pytest.approx(0.0, abs=0.02)

    def test_ece_positive(self):
        """ECE is non-negative for any inputs."""
        rng = np.random.default_rng(9)
        actuals = rng.uniform(2.0, 20.0, size=(30, 24))
        q_fc = np.sort(
            rng.uniform(2.0, 20.0, size=(30, len(QUANTILE_LEVELS), 24)), axis=1
        )
        ece = compute_ece(q_fc, actuals, QUANTILE_LEVELS)
        assert ece >= 0.0

    def test_ece_bounded_by_half(self):
        """Max possible ECE is 0.5 (integral of |q - 0| or |q - 1| over [0,1])."""
        rng = np.random.default_rng(5)
        actuals = rng.uniform(2.0, 20.0, size=(20, 48))
        q_fc = np.sort(
            rng.uniform(2.0, 20.0, size=(20, len(QUANTILE_LEVELS), 48)), axis=1
        )
        ece = compute_ece(q_fc, actuals, QUANTILE_LEVELS)
        assert ece <= 0.5

    def test_extreme_over_forecast_ece_large(self):
        """Forecasts always too high → ECE significantly > 0."""
        n_eps, fh = 50, 10
        actuals = np.ones((n_eps, fh)) * 3.0
        q_fc = np.ones((n_eps, len(QUANTILE_LEVELS), fh)) * 10.0
        ece = compute_ece(q_fc, actuals, QUANTILE_LEVELS)
        assert ece > 0.2
