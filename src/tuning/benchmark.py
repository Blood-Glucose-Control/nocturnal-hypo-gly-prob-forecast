import itertools
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
import pandas as pd
from src.data.data_loader import load_data
from src.data.data_cleaner import clean_data
import time

from src.tuning.param_grid import generate_param_grid
from src.utils.config_loader import load_yaml_config

from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

# Dictionary mapping names to forecaster classes
# Key is the key in the YMAL file and the value is the forecaster class
FORECASTER_REGISTRY = {
    "AutoARIMA": AutoARIMA,
    "ARIMA": ARIMA,
    "NaiveForecaster": NaiveForecaster,
    "ThetaForecaster": ThetaForecaster,
    "ExpSmoothing": ExponentialSmoothing,
}


def parse_output(benchmark, processed_output_dir, raw_output_dir):
    results_df = pd.read_csv(raw_output_dir)

    # Drop columns containing 'fold'
    results_df = results_df.loc[:, ~results_df.columns.str.contains("fold")]

    results_df.to_csv(processed_output_dir, index=False)

    return results_df


def load_diabetes_data(patient_id, df=None):
    if df is None:
        df = clean_data(load_data())
    patient_data = df[df["p_num"] == patient_id].copy()
    patient_data = patient_data.fillna(0)

    patient_data["time"] = pd.to_datetime(patient_data["time"], format="%H:%M:%S")

    y = patient_data["bg-0:00"]
    X = patient_data[["insulin-0:00", "carbs-0:00", "hr-0:00"]]

    return y, X


def get_dataset_loaders():
    # Create dataset loaders for each patient
    df = clean_data(load_data())
    patient_ids = df["p_num"].unique()

    dataset_loaders = []
    for patient_id in patient_ids:
        dataset_loaders.append(lambda p=patient_id: load_diabetes_data(p, df))
    return dataset_loaders


def get_cv_splitter(steps_per_hour, hours_to_forecast):
    cv_splitter = ExpandingWindowSplitter(
        initial_window=steps_per_hour,
        step_length=steps_per_hour,
        fh=steps_per_hour * hours_to_forecast,
    )

    return cv_splitter


def generate_estimators_from_param_grid():
    config = load_yaml_config("./src/tuning/configs/modset1.yaml")

    estimators = []
    for forecaster_name in config.keys():
        param_grid = generate_param_grid(forecaster_name, config)

        param_lists = []
        param_names = []

        for param_name, values in param_grid.items():
            param_names.append(param_name)
            param_lists.append(values)

        for param_values in itertools.product(*param_lists):
            param_dict = dict(zip(param_names, param_values))

            # Create identifier string
            param_str = "-".join(f"{k}_{str(v)}" for k, v in param_dict.items())
            estimator_id = f"{forecaster_name}-{param_str}"

            # Instantiate forecaster with parameters
            forecaster_class = FORECASTER_REGISTRY[forecaster_name]
            forecaster = forecaster_class(**param_dict)

            estimators.append((forecaster, estimator_id))

    return estimators


def get_benchmark(dataset_loaders, cv_splitter, scorers):
    benchmark = ForecastingBenchmark(
        backend="loky",  # Use parallel processing
        backend_params={"n_jobs": -1},  # Use all available CPU cores
    )

    # Generate all estimators
    estimators = generate_estimators_from_param_grid()
    for estimator, estimator_id in estimators:
        benchmark.add_estimator(estimator=estimator, estimator_id=estimator_id)

    for idx, dataset_loader in enumerate(dataset_loaders):
        benchmark.add_task(
            dataset_loader,
            cv_splitter,
            scorers,
            task_id=f"patient_{idx}",
        )
    return benchmark


def run_benchmark():
    # TODO: Remove this
    dataset_loaders = get_dataset_loaders()[:2]

    steps_per_hour = 12
    hours_to_forecast = 6
    cv_splitter = get_cv_splitter(steps_per_hour, hours_to_forecast)

    scorers = [
        PinballLoss(),
        MeanSquaredError(square_root=True),
    ]

    benchmark = get_benchmark(dataset_loaders, cv_splitter, scorers)

    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed_output_dir = f"./results/processed/{current_time}_forecasting_results.csv"
    raw_output_dir = f"./results/raw/{current_time}_forecasting_results.csv"

    benchmark.run(raw_output_dir)

    return parse_output(benchmark, processed_output_dir, raw_output_dir)


if __name__ == "__main__":
    start_time = time.time()
    run_benchmark()
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
