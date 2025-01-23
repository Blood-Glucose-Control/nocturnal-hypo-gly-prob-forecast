from scripts.quantile_forecast import (
    quantile_forecast_model_loop,
    get_quantile_forecasts,
)

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class TestQuantileForecast:
    default_params = {
        "BoxCoxBiasAdjustedForecaster": [{"forecaster": ARIMA}],
        "ColumnEnsembleForecaster": [{"forecaster": ARIMA}],
        "DartsLinearRegressionModel": [{"lags": [-1, -2, -3, -4, -5, -6]}],
    }

    skip_models = [
        "ARCH",
        "BoxCoxBiasAdjustedForecaster",
        "ColumnEnsembleForecaster",
        "ConformalIntervals",
        "ReducedRegressionForecaster",
        "DartsLinearRegressionModel",
        "DartsXGBModel",
        "DirRecTabularRegressionForecaster",
        "DirRecTimeSeriesRegressionForecaster",
        "DirectTabularRegressionForecaster",
        "DirectTimeSeriesRegressionForecaster",
        "MultioutputTabularRegressionForecaster",
        "MultioutputTimeSeriesRegressionForecaster",
        "RecursiveTabularRegressionForecaster",
        "RecursiveTimeSeriesRegressionForecaster",
        "DynamicFactor",
        "LTSFLinearForecaster",
        "LTSFDLinearForecaster",
        "LTSFNLinearForecaster",
        "LTSFTransformerForecaster",
        "SCINetForecaster",
        "CINNForecaster",
        "NeuralForecastRNN",
        "NeuralForecastLSTM",
        "PytorchForecastingTFT",
        "PytorchForecastingDeepAR",
        "PytorchForecastingNHiTS",
        "PytorchForecastingNBeats",
        "PyKANForecaster",
        "HFTransformersForecaster",
        "ChronosForecaster",
        "MOIRAIForecaster",
        "TimesFMForecaster",
        "TinyTimeMixerForecaster",
        "Croston",
        "EnsembleForecaster",
        "AutoEnsembleForecaster",
        "StackingForecaster",
        "ReconcilerForecaster",
        "OnlineEnsembleForecaster",
        "NormalHedgeEnsemble",
        "NNLSEnsemble",
        "UpdateEvery",
        "UpdateRefitsEvery",
        "DontUpdate",
        "HCrystalBallAdapter",
        "BaggingForecaster",
        "EnbPIForecaster",
        "FhPlexForecaster",
        "ForecastX",
        "ForecastingGridSearchCV",
        "ForecastingOptunaSearchCV",
        "TestQuantileForecast",
        "ForecastingPipeline",
        "ForecastingRandomizedSearchCV",
        "ForecastingSkoptSearchCV",
        "HierarchicalProphet",  # requires package 'prophetverse'
        "NaiveVariance",
        "Permute",
        "Prophetverse",  # needs the prophet package like above
        "SkforecastAutoreg",
        "SquaringResiduals",
        "StatsForecastAutoETS",
        "StatsForecastAutoTBATS",
        "TransformedTargetForecaster",
        "VAR",
        "VECM",
        "YfromX",
        "PolynomialTrendForecaster",  # this can't do quantile preds
        "Prophet",
        "SARIMAX",  # mreg and xreg matrices diff size
        "StatsForecastAutoARIMA",  # xreg should be float array
        "StatsModelsARIMA",  # endog and exog matrices diff size
    ]

    def test_model_loop(self):
        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(
            y_train,
            model_hyperparameters=self.default_params,
            skip_models=self.skip_models,
        )
        assert len(models) > 0

    def test_quantile_forecasts(self):
        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(
            y_train,
            model_hyperparameters=self.default_params,
            skip_models=self.skip_models,
        )
        y_test = pd.DataFrame({"y": [6, 7, 8, 9, 10]})
        quantile_forecasts = get_quantile_forecasts(
            models, y_test, forecast_horizon=[1, 2, 3, 4]
        )
        assert len(quantile_forecasts) > 0
from scripts.quantile_forecast import quantile_forecast_model_loop, get_quantile_forecasts

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class TestQuantileForecast:

    default_params = {
            "BoxCoxBiasAdjustedForecaster": [
                {
                    "forecaster": ARIMA
                }
            ],
            "ColumnEnsembleForecaster": [
                {
                    "forecaster": ARIMA
                }
            ],
            "DartsLinearRegressionModel": [
                {
                    "lags": [-1, -2, -3, -4, -5, -6]
                }
            ]
    }

    skip_models = [
        "ARCH",
        "BoxCoxBiasAdjustedForecaster",
        "ColumnEnsembleForecaster",
        "ConformalIntervals",
        "ReducedRegressionForecaster",
        "DartsLinearRegressionModel",
        "DartsXGBModel",
        "DirRecTabularRegressionForecaster",
        "DirRecTimeSeriesRegressionForecaster",
        "DirectTabularRegressionForecaster",
        "DirectTimeSeriesRegressionForecaster",
        "MultioutputTabularRegressionForecaster",
        "MultioutputTimeSeriesRegressionForecaster",
        "RecursiveTabularRegressionForecaster",
        "RecursiveTimeSeriesRegressionForecaster",
        "DynamicFactor",
        "LTSFLinearForecaster",
        "LTSFDLinearForecaster",
        "LTSFNLinearForecaster",
        "LTSFTransformerForecaster",
        "SCINetForecaster",
        "CINNForecaster",
        "NeuralForecastRNN",
        "NeuralForecastLSTM",
        "PytorchForecastingTFT",
        "PytorchForecastingDeepAR",
        "PytorchForecastingNHiTS",
        "PytorchForecastingNBeats",
        "PyKANForecaster",
        "HFTransformersForecaster",
        "ChronosForecaster",
        "MOIRAIForecaster",
        "TimesFMForecaster",
        "TinyTimeMixerForecaster",
        "Croston",
        "EnsembleForecaster",
        "AutoEnsembleForecaster",
        "StackingForecaster",
        "ReconcilerForecaster",
        "OnlineEnsembleForecaster",
        "NormalHedgeEnsemble",
        "NNLSEnsemble",
        "UpdateEvery",
        "UpdateRefitsEvery",
        "DontUpdate",
        "HCrystalBallAdapter",
        "BaggingForecaster",
        "EnbPIForecaster",
        "FhPlexForecaster",
        "ForecastX",
        "ForecastingGridSearchCV",
        "ForecastingOptunaSearchCV",
        "TestQuantileForecast",
        "ForecastingPipeline",
        "ForecastingRandomizedSearchCV",
        "ForecastingSkoptSearchCV",
        "HierarchicalProphet", # requires package 'prophetverse'
        'NaiveVariance',
        "Permute",
        "Prophetverse", # needs the prophet package like above
        "SkforecastAutoreg",
        "SquaringResiduals",
        "StatsForecastAutoETS",
        "StatsForecastAutoTBATS",
        "TransformedTargetForecaster",
        "VAR",
        "VECM",
        "YfromX",
        "PolynomialTrendForecaster", # this can't do quantile preds
        "Prophet",
        "SARIMAX", #mreg and xreg matrices diff size
        "StatsForecastAutoARIMA", #xreg should be float array
        "StatsModelsARIMA", #endog and exog matrices diff size
    ]

    def test_model_loop(self):
        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(y_train, model_hyperparameters=self.default_params, skip_models=self.skip_models)
        assert len(models) > 0

    def test_quantile_forecasts(self):

        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(y_train, model_hyperparameters=self.default_params, skip_models=self.skip_models)
        y_test = pd.DataFrame({"y": [6, 7, 8, 9, 10]})
        quantile_forecasts = get_quantile_forecasts(models, y_test, forecast_horizon=[1,2,3,4])
        assert len(quantile_forecasts) > 0