import itertools
from typing import Callable
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.registry import all_estimators
from sktime.transformations.series.impute import Imputer
import pandas as pd
from src.data.data_loader import load_data
from src.data.data_cleaner import clean_data

from src.tuning.param_grid import generate_param_grid
from src.utils.config_loader import load_yaml_config


forecasters = {name: est for name, est in all_estimators(estimator_types="forecaster")}


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


def impute_missing_values(
    df,
    columns,
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
) -> pd.DataFrame:
    """Imputes missing values in specified columns of a dataframe using different methods based on the data type.

    Args:
        df (pd.DataFrame): Input dataframe containing missing values
        columns (list): List of column names to impute missing values for
        bg_method (str, optional): Imputation method for blood glucose data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        hr_method (str, optional): Imputation method for heart rate data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        step_method (str, optional): Imputation method for step count data.
            Valid values: 'constant'.
        cal_method (str, optional): Imputation method for calorie data.
            Valid values: 'constant'.

    Returns:
        pd.DataFrame: Copy of input dataframe with missing values imputed using appropriate methods for each data type
    """
    df_imputed = df.copy()
    transform = None

    for col in columns:
        if col in df.columns:
            if "bg" in col.lower():
                transform = Imputer(method=bg_method)
            elif "hr" in col.lower():
                # Use linear or nearest neighbor interpolation for heart rate
                # TODO: Need more research on this
                transform = Imputer(method=hr_method)
            elif "step" in col.lower():
                # Use constant imputation with 0 for steps
                transform = Imputer(method=step_method, value=0)
            elif "cal" in col.lower():
                # Use constant imputation with minimum value for calories
                min_val = df[col].min()
                transform = Imputer(method=cal_method, value=min_val)

            if transform is not None:
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


def get_dataset_loaders(
    x_features,
    y_feature,
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
    n_patients=-1,
) -> list[Callable]:
    """Create the dataset, impute the x_features and y_feature, and create dataset loader functions for each patient.

    Args:
        x_features (list): List of feature column names to use as predictors.
        y_feature (str): Name of the target variable column to predict. (bg-0:00)
        bg_method (str, optional): Imputation method for blood glucose data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        hr_method (str, optional): Imputation method for heart rate data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        step_method (str, optional): Imputation method for step count data.
            Valid values: 'constant'.
        cal_method (str, optional): Imputation method for calorie data.
            Valid values: 'constant'.
        n_patients (int, optional): Number of patients to load.
            If -1, all patients are loaded.
    Returns:
        list[Callable]: List of dataset loader functions, one for each patient.
    """
    # Load and clean data
    df = clean_data(load_data())

    # TODO: Impute Missing values for each columns
    df = impute_missing_values(df, columns=x_features, bg_method=bg_method)
    df = impute_missing_values(
        df,
        columns=y_feature,
        hr_method=hr_method,
        step_method=step_method,
        cal_method=cal_method,
    )

    if n_patients == -1:
        n_patients = len(df["p_num"].unique())

    # Create dataset loaders for each patient
    patient_ids = df["p_num"].unique()
    dataset_loaders = []
    for patient_id in patient_ids[:n_patients]:
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


def load_model(model_name):
    """Loads the sktime model class given the model name"""
    if model_name not in forecasters:
        raise ValueError(f"Model {model_name} not found in sktime")

    ForecasterClass = forecasters[model_name]  # Get the class
    return ForecasterClass


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
            forecaster_class = load_model(forecaster_name)
            forecaster = forecaster_class(**param_dict)

            estimators.append((forecaster, estimator_id))

    return estimators


def get_benchmark(dataset_loaders, cv_splitter, scorers, ymal_path, cores_num=-1):
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
        backend_params={"n_jobs": cores_num},  # Use all available CPU cores
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


def run_benchmark(
    y_features=["bg-0:00"],
    x_features=["insulin-0:00", "carbs-0:00", "hr-0:00"],
    steps_per_hour=12,
    hours_to_forecast=6,
    ymal_path="./src/tuning/configs/modset1.yaml",
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
    processed_dir="./results/processed",
    raw_dir="./results/raw",
    cores_num=-1,
    n_patients=-1,
) -> None:
    """
    Run benchmarking experiments for diabetes forecasting models.
    Args:
        y_features (list[str], optional): Target variables to forecast. Defaults to ["bg-0:00"].
        x_features (list[str], optional): Input features for forecasting. Defaults to ["insulin-0:00", "carbs-0:00", "hr-0:00"].

        steps_per_hour (int, optional): Number of time steps per hour in the data. Defaults to 12.
        hours_to_forecast (int, optional): Number of hours to forecast ahead. Defaults to 6.

        ymal_path (str, optional): Path to YAML config file with model parameters. Defaults to "./src/tuning/configs/modset1.yaml".

        bg_method (str, optional): Imputation method for blood glucose. Defaults to "linear".
        hr_method (str, optional): Imputation method for heart rate. Defaults to "linear".
        step_method (str, optional): Imputation method for steps. Defaults to "constant".
        cal_method (str, optional): Imputation method for calories. Defaults to "constant".

        processed_dir (str, optional): Directory for processed results. Defaults to "./results/processed".
        raw_dir (str, optional): Directory for raw results. Defaults to "./results/raw".

        cores_num (int, optional): Number of CPU cores to use (-1 for all). Defaults to -1.
        n_patients (int, optional): Number of patients to include (-1 for all). Defaults to -1.

    The function:
    1. Loads and preprocesses patient datasets with specified imputation methods
    2. Configures cross-validation and scoring metrics
    3. Generates model configurations from YAML file
    4. Runs benchmarking experiments in parallel
    5. Saves raw and processed results to specified directories
    """
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed_output_dir = f"{processed_dir}/{current_time}_forecasting_results.csv"
    raw_output_dir = f"{raw_dir}/{current_time}_forecasting_results.csv"

    # Get dataset loaders with imputed missing values
    dataset_loaders = get_dataset_loaders(
        x_features=x_features,
        y_feature=y_features,
        bg_method=bg_method,
        hr_method=hr_method,
        step_method=step_method,
        cal_method=cal_method,
        n_patients=n_patients,
    )

    # ADD THE SCORERS HERE
    scorers = [
        PinballLoss(),
        MeanSquaredError(square_root=True),
    ]

    # Get the cross-validation splitter
    cv_splitter = get_cv_splitter(
        steps_per_hour=steps_per_hour, hours_to_forecast=hours_to_forecast
    )

    # Get the benchmark
    benchmark = get_benchmark(
        dataset_loaders=dataset_loaders,
        cv_splitter=cv_splitter,
        scorers=scorers,
        ymal_path=ymal_path,
        cores_num=cores_num,
    )

    # Run the benchmark
    benchmark.run(raw_output_dir)

    # Process the results
    parse_output(processed_output_dir, raw_output_dir)
