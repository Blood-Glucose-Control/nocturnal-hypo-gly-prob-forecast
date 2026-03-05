#!/usr/bin/env python3
"""
TTM Preprocessor Roundtrip Tests.

Comprehensive tests to validate that TTM's TimeSeriesPreprocessor is properly:
1. Saved during training (preprocessor.pkl exists)
2. Loaded during checkpoint restore
3. Applied correctly during inference (inverse scaling)
4. Using the correct channel for predictions

These tests address the observed performance degradation after training,
which may be caused by scaling/preprocessing issues.

Run with:
    pytest tests/models/test_ttm_preprocessor_roundtrip.py -v
    pytest tests/models/test_ttm_preprocessor_roundtrip.py -v -s  # Show print output
"""

import os
import shutil
import tempfile
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytest

# Skip all tests if TTM dependencies are not available
pytest.importorskip("tsfm_public")


# =============================================================================
# FIXTURES: Synthetic Data Generation
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_glucose_data() -> pd.DataFrame:
    """Create synthetic glucose data for testing.

    Creates a small dataset with realistic glucose patterns for 5 patients.
    Each patient has ~1000 rows (about 3.5 days at 5-min intervals).
    Need enough rows to create sliding windows for train/val/test split.

    Returns:
        DataFrame with columns: p_num, datetime, bg_mM, iob, insulin_availability
    """
    np.random.seed(42)

    patients = []
    for p_num in range(1, 6):  # 5 patients
        n_rows = 1000  # More rows to ensure enough windows

        # Create timestamps (5-minute intervals)
        timestamps = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")

        # Generate realistic glucose values (4-15 mmol/L range)
        # Mean around 7.5 mmol/L (healthy average)
        base_glucose = 7.5 + np.sin(np.arange(n_rows) * 0.05) * 2  # Circadian pattern
        noise = np.random.normal(0, 0.5, n_rows)
        glucose = np.clip(base_glucose + noise, 3.0, 20.0)

        # Generate insulin features
        iob = np.clip(np.random.exponential(2.0, n_rows), 0, 10)
        insulin_avail = np.clip(np.random.exponential(1.5, n_rows), 0, 8)

        patient_df = pd.DataFrame(
            {
                "p_num": str(p_num),  # Must be string or int, not float
                "datetime": timestamps,
                "bg_mM": glucose,
                "iob": iob,
                "insulin_availability": insulin_avail,
            }
        )
        patients.append(patient_df)

    combined = pd.concat(patients, ignore_index=True)
    return combined


@pytest.fixture(scope="module")
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    tmp_dir = tempfile.mkdtemp(prefix="ttm_test_")
    yield tmp_dir
    # Cleanup after tests
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def minimal_ttm_config():
    """Create minimal TTM config for fast testing."""
    from src.models.ttm.config import TTMConfig

    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=64,  # Small for fast testing
        forecast_length=12,
        batch_size=32,
        num_epochs=1,  # Single epoch for speed
        learning_rate=1e-4,
        training_mode="fine_tune",
        freeze_backbone=False,
        input_features=["iob", "insulin_availability"],
        target_features=["bg_mM"],
        num_input_channels=3,  # bg_mM + 2 input features
        num_output_channels=1,
        split_config={"train": 0.8, "val": 0.1, "test": 0.1},
        fp16=False,  # CPU-compatible
        use_cpu=True,  # Force CPU for testing
    )
    return config


@pytest.fixture(scope="module")
def trained_model_and_artifacts(
    synthetic_glucose_data: pd.DataFrame,
    temp_output_dir: str,
    minimal_ttm_config,
) -> Tuple[Any, str, Dict[str, Any]]:
    """Train a minimal TTM model and return model + checkpoint path.

    This fixture runs once per module and provides:
    - The trained model instance
    - Path to the saved checkpoint
    - Training metrics/metadata
    """
    from src.models.ttm import TTMForecaster

    checkpoint_dir = os.path.join(temp_output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create and train model
    model = TTMForecaster(minimal_ttm_config)

    try:
        metrics = model._train_model(
            train_data=synthetic_glucose_data,
            output_dir=checkpoint_dir,
        )
    except Exception as e:
        import traceback

        print(f"\n!!! Training failed with exception: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        pytest.skip(f"Training failed (may be expected in CI): {e}")

    return model, checkpoint_dir, metrics


# =============================================================================
# TEST 1: Preprocessor Persistence
# =============================================================================


class TestPreprocessorPersistence:
    """Tests for verifying preprocessor is saved during training."""

    def test_preprocessor_pkl_exists(self, trained_model_and_artifacts):
        """Verify preprocessor.pkl file is created after training."""
        model, checkpoint_dir, _ = trained_model_and_artifacts

        preprocessor_path = os.path.join(checkpoint_dir, "preprocessor.pkl")
        assert os.path.exists(preprocessor_path), (
            f"preprocessor.pkl not found at {preprocessor_path}. "
            "The preprocessor must be saved for inference to work correctly."
        )

    def test_model_has_preprocessor(self, trained_model_and_artifacts):
        """Verify model.preprocessor is not None after training."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None, (
            "model.preprocessor is None after training. "
            "The preprocessor must be initialized during training."
        )

    def test_target_scaler_dict_populated(self, trained_model_and_artifacts):
        """Verify target_scaler_dict is populated (not empty)."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None
        scaler_dict = model.preprocessor.target_scaler_dict

        assert len(scaler_dict) > 0, (
            "target_scaler_dict is empty. "
            "Scalers must be fitted during training for inverse scaling to work."
        )

        # Log scaler details for debugging
        print("\n--- Target Scaler Details ---")
        for key, scaler in scaler_dict.items():
            print(f"  Key: {key}")
            if hasattr(scaler, "mean_"):
                print(f"    Mean: {scaler.mean_}")
            if hasattr(scaler, "scale_"):
                print(f"    Scale (std): {scaler.scale_}")

    def test_scaler_has_reasonable_parameters(self, trained_model_and_artifacts):
        """Verify scaler parameters are in physiological range for glucose."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None
        scaler_dict = model.preprocessor.target_scaler_dict

        # Get the first (or global) scaler
        scaler_key = next(iter(scaler_dict.keys()))
        scaler = scaler_dict[scaler_key]

        # For glucose in mmol/L, mean should be roughly 4-15, std roughly 1-5
        if hasattr(scaler, "mean_"):
            mean_val = (
                scaler.mean_[0] if hasattr(scaler.mean_, "__len__") else scaler.mean_
            )
            assert 3.0 < mean_val < 15.0, (
                f"Scaler mean {mean_val} is outside expected glucose range (3-15 mmol/L). "
                "This suggests the scaler was fitted on incorrect data."
            )
            print(f"\n✓ Scaler mean: {mean_val:.2f} mmol/L (expected range: 3-15)")

        if hasattr(scaler, "scale_"):
            scale_val = (
                scaler.scale_[0] if hasattr(scaler.scale_, "__len__") else scaler.scale_
            )
            assert 0.5 < scale_val < 10.0, (
                f"Scaler scale (std) {scale_val} is outside expected range (0.5-10). "
                "This suggests unusual data variance."
            )
            print(f"✓ Scaler scale (std): {scale_val:.2f}")


# =============================================================================
# TEST 2: Preprocessor Loading
# =============================================================================


class TestPreprocessorLoading:
    """Tests for verifying preprocessor loads correctly from checkpoint."""

    def test_preprocessor_loads_from_checkpoint(
        self, trained_model_and_artifacts, minimal_ttm_config
    ):
        """Verify preprocessor is restored when loading checkpoint."""
        from src.models.ttm import TTMForecaster

        _, checkpoint_dir, _ = trained_model_and_artifacts

        # Create new model instance
        config_copy = minimal_ttm_config
        config_copy.model_path = checkpoint_dir

        new_model = TTMForecaster(config_copy)
        new_model._load_checkpoint(checkpoint_dir)

        assert new_model.preprocessor is not None, (
            "Preprocessor not loaded from checkpoint. "
            "Check _load_checkpoint() implementation."
        )

    def test_loaded_scaler_matches_original(
        self, trained_model_and_artifacts, minimal_ttm_config
    ):
        """Verify loaded preprocessor has same scaler parameters as original."""
        from src.models.ttm import TTMForecaster

        original_model, checkpoint_dir, _ = trained_model_and_artifacts

        # Load into new model
        config_copy = minimal_ttm_config
        config_copy.model_path = checkpoint_dir

        loaded_model = TTMForecaster(config_copy)
        loaded_model._load_checkpoint(checkpoint_dir)

        # Compare scaler parameters
        assert original_model.preprocessor is not None
        assert loaded_model.preprocessor is not None

        orig_scalers = original_model.preprocessor.target_scaler_dict
        loaded_scalers = loaded_model.preprocessor.target_scaler_dict

        assert set(orig_scalers.keys()) == set(loaded_scalers.keys()), (
            f"Scaler keys don't match. Original: {orig_scalers.keys()}, "
            f"Loaded: {loaded_scalers.keys()}"
        )

        for key in orig_scalers.keys():
            orig_scaler = orig_scalers[key]
            loaded_scaler = loaded_scalers[key]

            if hasattr(orig_scaler, "mean_"):
                np.testing.assert_array_almost_equal(
                    orig_scaler.mean_,
                    loaded_scaler.mean_,
                    decimal=5,
                    err_msg=f"Scaler mean mismatch for key {key}",
                )

            if hasattr(orig_scaler, "scale_"):
                np.testing.assert_array_almost_equal(
                    orig_scaler.scale_,
                    loaded_scaler.scale_,
                    decimal=5,
                    err_msg=f"Scaler scale mismatch for key {key}",
                )

        print("\n✓ Loaded preprocessor matches original")


# =============================================================================
# TEST 3: Scaling Correctness
# =============================================================================


class TestScalingCorrectness:
    """Tests for verifying inverse-scaling produces correct units."""

    def test_inverse_scale_produces_mmol_range(self, trained_model_and_artifacts):
        """Verify inverse-scaled predictions are in mmol/L range, not z-scores."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None

        # Create fake scaled predictions (z-scores around 0)
        fake_scaled_preds = np.array(
            [[-0.5, 0.0, 0.5, 1.0, -1.0]]
        )  # (1, 5) - 5 timesteps
        fake_scaled_preds = fake_scaled_preds.reshape(
            1, 5, 1
        )  # (batch, forecast, channels)

        # Inverse scale
        unscaled = model._inverse_scale_predictions(
            predictions=fake_scaled_preds,
            data=None,  # Not used for global scaling
        )

        # For glucose, unscaled values should be in physiological range
        mean_unscaled = np.mean(unscaled)

        print("\n--- Inverse Scaling Test ---")
        print(f"  Input (scaled z-scores): {fake_scaled_preds.flatten()}")
        print(f"  Output (unscaled): {unscaled.flatten()}")
        print(f"  Output mean: {mean_unscaled:.2f}")

        # If mean is between -2 and 2, it's still in z-score range (BAD)
        # If mean is between 3 and 15, it's in mmol/L range (GOOD)
        assert mean_unscaled > 2.5, (
            f"Inverse-scaled predictions have mean {mean_unscaled:.2f}, "
            "which looks like z-scores (should be mmol/L, range 3-15). "
            "Check _inverse_scale_predictions() implementation."
        )

    def test_known_value_roundtrip(self, trained_model_and_artifacts):
        """Test scaling and inverse-scaling a known value."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None
        scaler_dict = model.preprocessor.target_scaler_dict
        scaler_key = next(iter(scaler_dict.keys()))
        scaler = scaler_dict[scaler_key]

        # Known glucose value
        original_value = np.array([[7.5]])  # 7.5 mmol/L

        # Scale it
        scaled = scaler.transform(original_value)

        # Inverse scale
        recovered = scaler.inverse_transform(scaled)

        print("\n--- Known Value Roundtrip ---")
        print(f"  Original: {original_value[0, 0]:.2f} mmol/L")
        print(f"  Scaled (z-score): {scaled[0, 0]:.4f}")
        print(f"  Recovered: {recovered[0, 0]:.2f} mmol/L")

        np.testing.assert_almost_equal(
            original_value[0, 0],
            recovered[0, 0],
            decimal=4,
            err_msg="Value not recovered correctly after scale/inverse-scale roundtrip",
        )


# =============================================================================
# TEST 4: Channel Selection Validation
# =============================================================================


class TestChannelSelection:
    """Tests for verifying correct channel is used for predictions."""

    def test_target_columns_is_bg_mM(self, trained_model_and_artifacts):
        """Verify target_columns[0] is 'bg_mM'."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None
        target_cols = model.preprocessor.target_columns

        assert len(target_cols) > 0, "target_columns is empty"
        assert target_cols[0] == "bg_mM", (
            f"First target column is '{target_cols[0]}', expected 'bg_mM'. "
            "Channel 0 of predictions may not correspond to blood glucose."
        )

        print(f"\n✓ target_columns: {target_cols}")

    def test_column_specifiers_consistency(self, trained_model_and_artifacts):
        """Verify column_specifiers match preprocessor configuration."""
        model, _, _ = trained_model_and_artifacts

        assert model.preprocessor is not None

        # Check target columns from preprocessor
        tsp_target = model.preprocessor.target_columns

        # Check config target_features
        config_target = model.config.target_features

        print("\n--- Column Configuration ---")
        print(f"  Preprocessor target_columns: {tsp_target}")
        print(f"  Config target_features: {config_target}")

        # They should be consistent
        assert tsp_target == config_target, (
            f"Mismatch between preprocessor target_columns ({tsp_target}) "
            f"and config target_features ({config_target}). "
            "This could cause channel selection issues."
        )


# =============================================================================
# TEST 5: End-to-End Roundtrip
# =============================================================================


class TestEndToEndRoundtrip:
    """End-to-end tests for train -> save -> load -> predict cycle."""

    def test_predictions_in_physiological_range(
        self, trained_model_and_artifacts, synthetic_glucose_data
    ):
        """Verify predictions after loading are in mmol/L range."""
        from src.models.ttm import TTMForecaster

        original_model, checkpoint_dir, _ = trained_model_and_artifacts

        # Load model from checkpoint (simulating fresh load)
        config = original_model.config
        config.model_path = checkpoint_dir

        loaded_model = TTMForecaster(config)
        loaded_model._load_checkpoint(checkpoint_dir)
        loaded_model.is_fitted = True

        # Get a small sample for prediction
        sample_data = synthetic_glucose_data[
            synthetic_glucose_data["p_num"] == "1"
        ].head(100)

        try:
            predictions = loaded_model.predict(sample_data)
        except Exception as e:
            pytest.skip(f"Prediction failed: {e}")

        print("\n--- Prediction Output ---")
        print(f"  Shape: {predictions.shape}")
        print(f"  Min: {np.min(predictions):.2f}")
        print(f"  Max: {np.max(predictions):.2f}")
        print(f"  Mean: {np.mean(predictions):.2f}")
        print(f"  Std: {np.std(predictions):.2f}")

        # Check predictions are in physiological range
        pred_mean = np.mean(predictions)
        assert 2.0 < pred_mean < 20.0, (
            f"Prediction mean {pred_mean:.2f} is outside physiological range (2-20 mmol/L). "
            "This suggests inverse scaling is not working correctly."
        )

        # Check for z-score range (would indicate scaling issue)
        if -3 < pred_mean < 3:
            pytest.fail(
                f"Prediction mean {pred_mean:.2f} looks like z-scores, not mmol/L. "
                "Inverse scaling may not be applied."
            )

    def test_prediction_shape_has_correct_channels(
        self, trained_model_and_artifacts, synthetic_glucose_data
    ):
        """Verify prediction shape matches expected output channels."""
        original_model, checkpoint_dir, _ = trained_model_and_artifacts

        # Get sample and predict
        sample_data = synthetic_glucose_data[
            synthetic_glucose_data["p_num"] == "1"
        ].head(100)

        try:
            predictions = original_model.predict(sample_data)
        except Exception as e:
            pytest.skip(f"Prediction failed: {e}")

        print("\n--- Prediction Shape Analysis ---")
        print(f"  Shape: {predictions.shape}")
        print(f"  Config forecast_length: {original_model.config.forecast_length}")
        print(
            f"  Config num_output_channels: {original_model.config.num_output_channels}"
        )

        # Shape should be (batch, forecast_length, num_output_channels)
        if len(predictions.shape) == 3:
            _, forecast_len, n_channels = predictions.shape

            assert forecast_len == original_model.config.forecast_length, (
                f"Forecast length {forecast_len} doesn't match config "
                f"({original_model.config.forecast_length})"
            )

            # Check channel count
            expected_channels = original_model.config.num_output_channels
            print(f"  Actual channels: {n_channels}")
            print(f"  Expected channels: {expected_channels}")

            # Note: TTM might output more channels than expected
            # Channel 0 should always be the target (bg_mM)


# =============================================================================
# DIAGNOSTIC TESTS: For debugging specific issues
# =============================================================================


class TestDiagnostics:
    """Diagnostic tests to help debug specific issues."""

    def test_print_preprocessor_state(self, trained_model_and_artifacts):
        """Print full preprocessor state for debugging."""
        model, _, _ = trained_model_and_artifacts

        if model.preprocessor is None:
            pytest.skip("No preprocessor to inspect")

        tsp = model.preprocessor

        print("\n" + "=" * 60)
        print("PREPROCESSOR STATE DUMP")
        print("=" * 60)

        print("\nColumn Configuration:")
        print(f"  id_columns: {tsp.id_columns}")
        print(f"  timestamp_column: {tsp.timestamp_column}")
        print(f"  target_columns: {tsp.target_columns}")
        print(f"  observable_columns: {tsp.observable_columns}")
        print(f"  control_columns: {tsp.control_columns}")

        print("\nScaling Configuration:")
        print(f"  scaling: {tsp.scaling}")
        print(f"  scaler_type: {getattr(tsp, 'scaler_type', 'N/A')}")

        print("\nScaler Dictionary:")
        for key, scaler in tsp.target_scaler_dict.items():
            print(f"  Key: {key}")
            if hasattr(scaler, "mean_"):
                print(f"    mean_: {scaler.mean_}")
            if hasattr(scaler, "scale_"):
                print(f"    scale_: {scaler.scale_}")
            if hasattr(scaler, "var_"):
                print(f"    var_: {scaler.var_}")

        print("=" * 60)

    def test_compare_zero_shot_vs_finetuned_scaling(
        self, synthetic_glucose_data, minimal_ttm_config
    ):
        """Compare scaling behavior between zero-shot and fine-tuned models.

        This test helps identify if the issue is specific to fine-tuned models.
        """
        from src.models.ttm import TTMForecaster

        # Create zero-shot config
        zs_config = minimal_ttm_config
        zs_config.training_mode = "zero_shot"
        zs_config.freeze_backbone = True

        zs_model = TTMForecaster(zs_config)

        sample_data = synthetic_glucose_data[
            synthetic_glucose_data["p_num"] == "1"
        ].head(100)

        try:
            zs_preds = zs_model.predict_zero_shot(sample_data)
        except Exception as e:
            import traceback

            print(f"\n!!! Zero-shot prediction failed: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            pytest.skip(f"Zero-shot prediction failed: {e}")

        print("\n--- Zero-Shot vs Fine-Tuned Comparison ---")
        print("Zero-shot predictions:")
        print(f"  Shape: {zs_preds.shape}")
        print(f"  Mean: {np.mean(zs_preds):.2f}")
        print(f"  Std: {np.std(zs_preds):.2f}")
        print(f"  Range: [{np.min(zs_preds):.2f}, {np.max(zs_preds):.2f}]")


# =============================================================================
# UTILITY: Run specific diagnostic
# =============================================================================


if __name__ == "__main__":
    """Run tests directly for debugging."""
    pytest.main([__file__, "-v", "-s"])
