from scripts.quantile_forecast import quantile_forecast_model_loop, get_quantile_forecasts

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class TestQuantileForecast:

    def test_model_loop(self):
        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(y_train)
        assert len(models) > 0

    def test_quantile_forecasts(self):

        model_hyperparameters = {
            "BoxCoxBiasAdjustedForecaster": [
                {
                    "forecaster": ARIMA
                }
            ]
        }

        y_train = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        models = quantile_forecast_model_loop(y_train, model_hyperparameters=model_hyperparameters)
        y_test = pd.DataFrame({"y": [6, 7, 8, 9, 10]})
        quantile_forecasts = get_quantile_forecasts(models, y_test)
        assert len(quantile_forecasts) > 0