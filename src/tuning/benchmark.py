import itertools
from typing import Callable
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
from sktime.transformations.series.impute import Imputer

# Dictionary mapping names to forecaster classes
# Key is the key in the YMAL file and the value is the forecaster class
FORECASTER_REGISTRY = {
    "AutoARIMA": AutoARIMA,
    "ARIMA": ARIMA,
    "NaiveForecaster": NaiveForecaster,
    "ThetaForecaster": ThetaForecaster,
    "ExpSmoothing": ExponentialSmoothing,
}


def parse_output(processed_output_dir, raw_output_dir) -> pd.DataFrame:
    """Get the results from the raw output directory, filter out the fold columns, and save the processed results.

    Args:
        processed_output_dir (str): Path to save the processed results CSV
        raw_output_dir (str): Path to the raw results CSV file

    Returns:
        pd.DataFrame: Processed results dataframe with fold columns removed
    """
    results_df = pd.read_csv(raw_output_dir)

    # Drop columns containing 'fold'
    results_df = results_df.loc[:, ~results_df.columns.str.contains("fold")]
    results_df.to_csv(processed_output_dir, index=False)

    return results_df


def impute_missing_values(df, columns_to_impute, method="ffill") -> pd.DataFrame:
    """Imputes missing values in specified columns of a dataframe.
        Probably won't work with categorical features.

    Args:
        df (pd.DataFrame): Input dataframe containing missing values
        columns_to_impute (list): List of column names to impute missing values for
        method (str, optional): Imputation method to use.
            Valid values are: 'drift', 'linear', 'nearest', 'constant', 'mean', 'median', 'bfill', 'ffill', 'random'. Defaults to "ffill".

    Returns:
        pd.DataFrame: Copy of input dataframe with missing values imputed
    """
    df_imputed = df.copy()

    transform = Imputer(method=method)
    for col in columns_to_impute:
        if col in df.columns:
            df_imputed[col] = transform.fit_transform(df[col].to_frame())

    return df_imputed


def load_diabetes_data(patient_id, df=None, y_feature=None, x_features=[]):
    """Loads and preprocesses diabetes data for a specific patient. This function is intented to be called by the benchmark.

    Args:
        patient_id: ID of the patient to load data for
        df (pd.DataFrame, optional): Pre-loaded dataframe to use. If None, loads and cleans data.
        y_feature (str): Name of target variable column to predict
        x_features (list): List of feature column names to use as predictors

    Returns:
        tuple: (y, X) where y is the target series and X is a DataFrame of predictor features

    Raises:
        ValueError: If y_feature is None or dataframe is not provided
    """
    if y_feature is None:
        raise ValueError(
            "y_feature cannot be None - must specify target variable for forecasting"
        )
    if df is None:
        raise ValueError("dataframe must be provided")

    patient_data = df[df["p_num"] == patient_id].copy()
    patient_data["time"] = pd.to_datetime(patient_data["time"], format="%H:%M:%S")

    y = patient_data[y_feature]
    X = patient_data[x_features]

    return y, X


def get_dataset_loaders(x_features, y_feature, impute_method="ffill") -> list[Callable]:
    """Creates dataset loader functions for each patient in the diabetes dataset and imputes missing values.

    Args:
        x_features (list): List of feature column names to use as predictors.
        y_feature (str): Name of the target variable column to predict.

    Returns:
        list[Callable]: List of dataset loader functions, one for each patient.
    """
    # Create dataset loaders for each patient
    df = clean_data(load_data())
    patient_ids = df["p_num"].unique()

    # TODO: Won't work with categorical features, probably will need a seperate list of x_features we want to impuate and combine them later
    df = impute_missing_values(df, columns_to_impute=x_features, method=impute_method)
    df = impute_missing_values(df, columns_to_impute=y_feature, method=impute_method)

    dataset_loaders = []
    for patient_id in patient_ids:
        dataset_loaders.append(
            lambda p=patient_id: load_diabetes_data(p, df, y_feature, x_features)
        )
    return dataset_loaders


def get_cv_splitter(steps_per_hour, hours_to_forecast) -> ExpandingWindowSplitter:
    """Creates an expanding window cross-validation splitter for time series forecasting.

    Args:
        steps_per_hour (int): Number of time steps per hour in the data
        hours_to_forecast (int): Number of hours to forecast ahead

    Returns:
        ExpandingWindowSplitter: Cross-validation splitter configured with:
            - Initial training window of steps_per_hour
            - Step size of steps_per_hour between splits
            - Forecast horizon of steps_per_hour * hours_to_forecast
    """
    cv_splitter = ExpandingWindowSplitter(
        initial_window=steps_per_hour,
        step_length=steps_per_hour,
        fh=steps_per_hour * hours_to_forecast,
    )

    return cv_splitter


def generate_estimators_from_param_grid(ymal_path) -> list[tuple[Callable, str]]:
    """Generates a list of forecasting estimators with different parameter combinations.
       forecaster_name is the key in the YAML file and the value is the forecaster class.

    Args:
        ymal_path (str): Path to YAML config file containing forecaster parameters.

    Returns:
        list[tuple[Callable, str]]: List of tuples containing (forecaster instance, estimator_id string).
            The forecaster instance is initialized with parameters from the config.
            The estimator_id uniquely identifies the forecaster and its parameters.
    """
    config = load_yaml_config(ymal_path)

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


def get_benchmark(dataset_loaders, cv_splitter, scorers, ymal_path):
    """Creates and configures a ForecastingBenchmark instance for evaluating multiple forecasting models.

    Args:
        dataset_loaders (list[Callable]): List of functions that load individual patient datasets
        cv_splitter (ExpandingWindowSplitter): Cross-validation splitter that defines train/test splits
        scorers: List of scoring metrics to evaluate forecasting performance
        ymal_path (str): Path to YAML config file containing model parameters to test

    Returns:
        ForecastingBenchmark: Configured benchmark object ready to run experiments
    """
    benchmark = ForecastingBenchmark(
        backend="loky",  # Use parallel processing
        backend_params={"n_jobs": -1},  # Use all available CPU cores
    )

    # Generate all estimators
    estimators = generate_estimators_from_param_grid(ymal_path)
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


def run_benchmark() -> None:
    # Variables:
    y_features = ["bg-0:00"]
    x_features = ["insulin-0:00", "carbs-0:00", "hr-0:00"]
    steps_per_hour = 12
    hours_to_forecast = 6
    ymal_path = "./src/tuning/configs/modset1.yaml"

    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed_output_dir = f"./results/processed/{current_time}_forecasting_results.csv"
    raw_output_dir = f"./results/raw/{current_time}_forecasting_results.csv"

    # Get dataset loaders with imputed missing values
    dataset_loaders = get_dataset_loaders(x_features, y_features)[:2]

    cv_splitter = get_cv_splitter(steps_per_hour, hours_to_forecast)

    # ADD THE SCORERS HERE
    scorers = [
        PinballLoss(),
        MeanSquaredError(square_root=True),
    ]

    benchmark = get_benchmark(dataset_loaders, cv_splitter, scorers, ymal_path)

    benchmark.run(raw_output_dir)

    parse_output(processed_output_dir, raw_output_dir)


if __name__ == "__main__":
    start_time = time.time()
    run_benchmark()
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
