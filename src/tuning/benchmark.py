import itertools
import time
from typing import Callable
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.forecasting.compose import FallbackForecaster
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import ExpandingSlidingWindowSplitter, ExpandingWindowSplitter
from sktime.split.base import BaseWindowSplitter
from sktime.transformations.series.impute import Imputer
import pandas as pd
import numpy as np
import os
import json
from src.data.data_loader import load_data, get_train_validation_split

from src.tuning.param_grid import generate_param_grid
from src.utils.config_loader import load_yaml_config
from src.tuning.load_estimators import (
    load_all_forecasters,
    load_all_regressors,
    get_estimator,
)


forecasters = load_all_forecasters()
regressors = load_all_regressors()

# Global variables
run_config = {
    "timestamp": "",
    "yaml_path": "",
    "interval": "",
    "patient_numbers": [],
    "scorers": {},
    "cv_type": {},
    "models_count": [],
    "impute_methods": {},
    "time_taken": 0,
    "x_features": [],
    "y_features": [],
    "initial_window": 0,
    "step_length": 0,
    "steps_per_hour": 0,
    "hours_to_forecast": 0,
}


def parse_output(
    current_time, yaml_name, raw_output_dir, processed_output_dir, scorers
) -> pd.DataFrame:
    """Get the results from the raw output directory, filter out the fold columns, and save the processed results.

    Args:
        processed_output_dir (str): Path to save the processed results CSV
        raw_output_dir (str): Path to the raw results CSV file

    Returns:
        pd.DataFrame: Processed results dataframe with fold columns removed
    """
    results_df = pd.read_csv(raw_output_dir)

    model_name = yaml_name.split("_")[1]
    interval = yaml_name.split("_")[-1] + "s"
    # Drop columns containing 'fold'
    results_df = results_df.loc[:, ~results_df.columns.str.contains("fold")]
    for scorer in scorers:
        # Save results to {processed_output_dir}/{scorer_name}/{interval}/{model_name}.csv
        scorer_dir = os.path.join(
            processed_output_dir,
            (scorer.__class__.__name__).lower(),
            model_name,
            interval,
        )
        os.makedirs(scorer_dir, exist_ok=True)
        # Keep only validation_id, model_id, runtime_secs and scorer columns
        keep_cols = ["validation_id", "model_id", "runtime_secs"] + [
            col for col in results_df.columns if scorer.__class__.__name__ in col
        ]
        score_df = results_df[keep_cols]
        score_df.to_csv(
            os.path.join(scorer_dir, f"{current_time}_{model_name}.csv"), index=False
        )

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


def get_patient_ids(df, is_5min, n_patients=-1):
    """Get the patient ids from the dataframe based on the time interval and number of patients."""
    patient_ids = df["p_num"].unique()
    selected_patients = []

    for patient_id in patient_ids:
        patient_data = df[df["p_num"] == patient_id]
        expected_interval = pd.Timedelta(minutes=5 if is_5min else 15)
        # Use the first two rows to determine the time interval
        time_diff = pd.to_datetime(patient_data["time"].iloc[1]) - pd.to_datetime(
            patient_data["time"].iloc[0]
        )

        if time_diff == expected_interval:
            selected_patients.append(patient_id)

    n_patients = min(n_patients, len(selected_patients))
    interval = "5-min" if is_5min else "15-min"

    print(
        f"Loading {n_patients} patients with {interval} interval: {selected_patients[:n_patients]}"
    )
    run_config["interval"] = interval
    run_config["patient_numbers"] = selected_patients[:n_patients]

    return selected_patients[:n_patients]


def get_dataset_loaders(
    x_features,
    y_feature,
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
    is_5min=True,
    n_patients=-1,
) -> dict[str, Callable]:
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
        is_5min (bool): Whether to use patients with 5-min interval.
            if true, load 5-min patient, else load 15-min patients
        n_patients (int, optional): Number of patients to load.
            If -1, all patients with given interval are loaded.
    Returns:
        dict[str, Callable]: Dictionary of dataset loader functions, one for each patient.
    """
    # Load and clean data
    df = load_data(use_cached=True)
    df, _ = get_train_validation_split(df)

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
    patient_ids = get_patient_ids(df, is_5min, n_patients)

    # Create dictionary of dataset loaders mapping patient_id to dataset loader function so we can get the correct patient_id in the benchmark
    dataset_loaders = {}
    for patient_id in patient_ids[:n_patients]:
        dataset_loaders[patient_id] = lambda p=patient_id: load_diabetes_data(
            patient_id=p, df=df, y_feature=y_feature, x_features=x_features
        )
    return dataset_loaders


def get_cv_splitter(
    initial_window, step_length, steps_per_hour, hours_to_forecast, cv_type="expanding"
) -> BaseWindowSplitter:
    """Creates an expanding window cross-validation splitter for time series forecasting.

    Args:
        initial_window (int): Initial training window size in number of time steps
        step_length (int): Step size between training windows in number of time steps
        steps_per_hour (int): Number of time steps per hour in the data
        hours_to_forecast (int): Number of hours to forecast ahead

    Returns:
        ExpandingWindowSplitter: Cross-validation splitter configured with:
            - Initial training window of steps_per_hour
            - Step size of steps_per_hour between splits
            - Forecast horizon of steps_per_hour * hours_to_forecast
    """

    if cv_type == "expanding":
        cv_splitter = ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=np.arange(1, steps_per_hour * hours_to_forecast + 1),
        )
    elif cv_type == "expanding_sliding":
        cv_splitter = ExpandingSlidingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=np.arange(1, steps_per_hour * hours_to_forecast + 1),
        )

    else:
        raise ValueError(f"Invalid cv_type: {cv_type}")

    run_config["cv_type"] = cv_splitter.__class__.__name__
    run_config["initial_window"] = initial_window
    run_config["step_length"] = step_length
    run_config["steps_per_hour"] = steps_per_hour
    run_config["hours_to_forecast"] = hours_to_forecast

    return cv_splitter


def load_model(model_name):
    """Loads the sktime model class given the model name"""
    if model_name not in forecasters:
        raise ValueError(f"Model {model_name} not found in sktime")

    ForecasterClass = get_estimator(forecasters, model_name)  # Get the class
    return ForecasterClass


def generate_estimators_from_param_grid(yaml_path) -> list[tuple[Callable, str]]:
    """Generates a list of forecasting estimators with different parameter combinations.
       forecaster_name is the key in the YAML file and the value is the forecaster class.

    Args:
        yaml_path (str): Path to YAML config file containing forecaster parameters.

    Returns:
        list[tuple[Callable, str]]: List of tuples containing (forecaster instance, estimator_id string).
            The forecaster instance is initialized with parameters from the config.
            The estimator_id uniquely identifies the forecaster and its parameters.
    """
    config = load_yaml_config(yaml_path)
    run_config["yaml_content"] = config

    estimators = []
    for forecaster_name in config.keys():
        param_grid = generate_param_grid(forecaster_name, config)

        param_lists = []
        param_names = []
        count = 0

        for param_name, values in param_grid.items():
            param_names.append(param_name)
            param_lists.append(values)

        for param_values in itertools.product(*param_lists):
            param_dict = dict(zip(param_names, param_values))

            # Create identifier string
            param_str = "-".join(f"{k}_{str(v)}" for k, v in param_dict.items())
            estimator_id = f"{forecaster_name}-{param_str}"

            # Fallback to native forecaster if the main forecaster fails
            forecaster = FallbackForecaster(
                forecasters=[
                    load_model(forecaster_name)(**param_dict),
                    load_model("NaiveForecaster")(),
                ],
                verbose=True,
            )

            estimators.append((forecaster, estimator_id))
            count += 1

        print(f"Training {count} {forecaster_name} models with different parameters")
        run_config["models_count"].append({forecaster_name: count})

    return estimators


def get_benchmark(dataset_loaders, cv_splitter, scorers, yaml_path, cores_num=-1):
    """Creates and configures a ForecastingBenchmark instance for evaluating multiple forecasting models.

    Args:
        dataset_loaders (dict[str, Callable]): Dictionary of functions that load individual patient datasets: {patient_id: dataset_loader}
        cv_splitter (ExpandingWindowSplitter): Cross-validation splitter that defines train/test splits
        scorers: List of scoring metrics to evaluate forecasting performance
        yaml_path (str): Path to YAML config file containing model parameters to test

    Returns:
        ForecastingBenchmark: Configured benchmark object ready to run experiments
    """
    benchmark = ForecastingBenchmark(
        backend="loky",  # Use parallel processing
        backend_params={"n_jobs": cores_num},  # Use all available CPU cores
    )

    # Generate all estimators
    estimators = generate_estimators_from_param_grid(yaml_path)
    for estimator, estimator_id in estimators:
        benchmark.add_estimator(estimator=estimator, estimator_id=estimator_id)

    for patient_id, dataset_loader in dataset_loaders.items():
        benchmark.add_task(
            dataset_loader,
            cv_splitter,
            scorers,
            task_id=patient_id,
            error_score="raise",
        )
    return benchmark


def save_config(run_config, config_dir, timestamp):
    """Saves the run config to a JSON file."""
    with open(f"{config_dir}/{timestamp}_run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)


def save_init_config(
    current_time: str,
    yaml_path: str,
    x_features: list[str],
    y_features: list[str],
    bg_method: str,
    hr_method: str,
    step_method: str,
    cal_method: str,
    description: str,
) -> None:
    run_config["timestamp"] = current_time
    run_config["yaml_path"] = yaml_path
    run_config["x_features"] = x_features
    run_config["y_features"] = y_features
    run_config["impute_methods"] = {
        "bg_method": bg_method,
        "hr_method": hr_method,
        "step_method": step_method,
        "cal_method": cal_method,
    }
    run_config["description"] = description


def run_benchmark(
    y_features=["bg-0:00"],
    x_features=["iob", "cob"],
    cv_type="expanding",
    initial_cv_window=12 * 24 * 3,
    cv_step_length=12 * 24 * 3,
    steps_per_hour=12,
    hours_to_forecast=6,
    yaml_path="./src/tuning/configs/old/modset1.yaml",
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
    processed_dir="./results/processed",
    raw_dir="./results/raw",
    config_dir="./results/configs",
    cores_num=-1,
    is_5min=True,
    n_patients=-1,
    description="No description is provided for this run",
) -> None:
    """
    Run benchmarking experiments for diabetes forecasting models. Constants columns will not work with some forecasting models.
    Args:
        y_features (list[str], optional): Target variables to forecast. Defaults to ["bg-0:00"].
        x_features (list[str], optional): Input features for forecasting. Defaults to ["insulin-0:00", "carbs-0:00", "hr-0:00"].

        steps_per_hour (int, optional): Number of time steps per hour in the data. Defaults to 12.
        hours_to_forecast (int, optional): Number of hours to forecast ahead. Defaults to 6.

        yaml_path (str, optional): Path to YAML config file with model parameters. Defaults to "./src/tuning/configs/modset1.yaml".

        bg_method (str, optional): Imputation method for blood glucose. Defaults to "linear".
        hr_method (str, optional): Imputation method for heart rate. Defaults to "linear".
        step_method (str, optional): Imputation method for steps. Defaults to "constant".
        cal_method (str, optional): Imputation method for calories. Defaults to "constant".

        processed_dir (str, optional): Directory for processed results. Defaults to "./results/processed".
        raw_dir (str, optional): Directory for raw results. Defaults to "./results/raw".

        cores_num (int, optional): Number of CPU cores to use (-1 for all). Defaults to -1.
        is_5min (bool, optional): Whether the data is 5 minutes apart. Defaults to True.
        n_patients (int, optional): Number of patients to include from the given interval patients (-1 for all). Defaults to -1.

    The function:
    1. Loads and preprocesses patient datasets with specified imputation methods
    2. Configures cross-validation and scoring metrics
    3. Generates model configurations from YAML file
    4. Runs benchmarking experiments in parallel
    5. Saves raw and processed results to specified directories
    """
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    yaml_name = yaml_path.split("/")[-1].replace(".yaml", "")
    raw_output_dir = f"{raw_dir}/{current_time}_{yaml_name}.csv"

    save_init_config(
        current_time=current_time,
        yaml_path=yaml_path,
        x_features=x_features,
        y_features=y_features,
        bg_method=bg_method,
        hr_method=hr_method,
        step_method=step_method,
        cal_method=cal_method,
        description=description,
    )

    # Get dataset loaders with imputed missing values
    dataset_loaders = get_dataset_loaders(
        x_features=x_features,
        y_feature=y_features,
        bg_method=bg_method,
        hr_method=hr_method,
        step_method=step_method,
        cal_method=cal_method,
        n_patients=n_patients,
        is_5min=is_5min,
    )

    # ADD THE SCORERS HERE
    scorers = [PinballLoss(), MeanSquaredError(square_root=True)]
    run_config["scorers"] = [scorer.__class__.__name__ for scorer in scorers]

    # Get the cross-validation splitter
    cv_splitter = get_cv_splitter(
        initial_window=initial_cv_window,
        step_length=cv_step_length,
        steps_per_hour=steps_per_hour,
        hours_to_forecast=hours_to_forecast,
        cv_type=cv_type,
    )

    # Get the benchmark
    benchmark = get_benchmark(
        dataset_loaders=dataset_loaders,
        cv_splitter=cv_splitter,
        scorers=scorers,
        yaml_path=yaml_path,
        cores_num=cores_num,
    )

    # Run the benchmark with timer
    start_time = time.time()
    print(json.dumps(run_config, indent=4))

    try:
        benchmark.run(raw_output_dir)
    except Exception as e:
        run_config["error"] = str(e)
        print(f"Error: {e}")
        print(f"Benchmark failed after {time.time() - start_time:.2f} seconds")
        return

    end_time = time.time()
    run_config["time_taken"] = end_time - start_time
    save_config(run_config, config_dir, current_time)

    # Process the results
    parse_output(current_time, yaml_name, raw_output_dir, processed_dir, scorers)
