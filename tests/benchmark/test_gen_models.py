import pytest
from unittest.mock import patch
from src.tuning.benchmark import generate_estimators_from_param_grid
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import FallbackForecaster


@pytest.fixture
def mock_config():
    return {
        "NaiveForecaster": {
            "strategy": ["mean", "last"],
            "sp": [0, 1],
            "window_length": [0, 1, 2],
        }
    }


@pytest.fixture
def mock_param_grid():
    return {"strategy": ["mean", "last"], "sp": [0, 1], "window_length": [0, 1, 2]}


# Need to add more combination for this
def test_generate_estimators_from_param_grid(mock_config, mock_param_grid):
    with (
        patch("src.tuning.benchmark.load_yaml_config") as mock_load_yaml,
        patch("src.tuning.benchmark.generate_param_grid") as mock_gen_params,
        patch("src.tuning.benchmark.load_model") as mock_load_model,
        patch("src.tuning.benchmark.run_config", {"models_count": []}),
    ):
        mock_load_yaml.return_value = mock_config
        mock_gen_params.return_value = mock_param_grid
        mock_load_model.side_effect = (
            lambda x: NaiveForecaster if x == "NaiveForecaster" else FallbackForecaster
        )

        estimators = generate_estimators_from_param_grid("dummy_path.yaml")

        expected_count = 12
        # Check if the estimator IDs are correctly formatted
        expected_ids = [
            "NaiveForecaster-strategy_mean-sp_0-window_length_0",
            "NaiveForecaster-strategy_mean-sp_0-window_length_1",
            "NaiveForecaster-strategy_mean-sp_0-window_length_2",
            "NaiveForecaster-strategy_mean-sp_1-window_length_0",
            "NaiveForecaster-strategy_mean-sp_1-window_length_1",
            "NaiveForecaster-strategy_mean-sp_1-window_length_2",
            "NaiveForecaster-strategy_last-sp_0-window_length_0",
            "NaiveForecaster-strategy_last-sp_0-window_length_1",
            "NaiveForecaster-strategy_last-sp_0-window_length_2",
            "NaiveForecaster-strategy_last-sp_1-window_length_0",
            "NaiveForecaster-strategy_last-sp_1-window_length_1",
            "NaiveForecaster-strategy_last-sp_1-window_length_2",
        ]

        assert len(estimators) == expected_count

        # All forecaster should be of type fallback forecaster
        actual_ids_and_types = [
            (est_id, isinstance(clsname, FallbackForecaster))
            for clsname, est_id in estimators
        ]
        for _, is_fallback in actual_ids_and_types:
            assert is_fallback, "Not all estimators are FallbackForecasters"

        actual_ids = [est_id for est_id, _ in actual_ids_and_types]
        assert sorted(actual_ids) == sorted(expected_ids)

        mock_load_yaml.assert_called_once_with("dummy_path.yaml")
        mock_gen_params.assert_called_once_with("NaiveForecaster", mock_config)
        # One for actual model and one for fallback forecaster
        assert mock_load_model.call_count == expected_count * 2
