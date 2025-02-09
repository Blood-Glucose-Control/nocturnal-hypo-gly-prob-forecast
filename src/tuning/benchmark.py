from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
import pandas as pd
from src.data.data_loader import load_data
from src.data.data_cleaner import clean_data
import time

# TODO:
# 1. Create a function that reads the ymal file
# 2. Add other features


def parse_output(benchmark):
    # Get current datetime
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_output_dir = f"./results/raw/{current_time}_forecasting_results.csv"
    processed_output_dir = f"./results/processed/{current_time}_forecasting_results.csv"

    benchmark.run(raw_output_dir)

    # Load the forecasting results from CSV
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

    # Set datetime index
    patient_data["time"] = pd.to_datetime(patient_data["time"])

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


# TODO: Make this a parameter
def get_cv_splitter():
    steps_per_hour = 12
    hours_to_forecast = 6
    cv_splitter = ExpandingWindowSplitter(
        initial_window=steps_per_hour,
        step_length=steps_per_hour,
        fh=steps_per_hour * hours_to_forecast,
    )
    return cv_splitter


def get_benchmark(dataset_loaders, cv_splitter, scorers):
    benchmark = ForecastingBenchmark()

    # TODO: Read estimator and their strageties from a file
    benchmark.add_estimator(
        estimator=NaiveForecaster(strategy="mean", sp=1),
        estimator_id="NaiveForecaster-mean",
    )
    benchmark.add_estimator(
        estimator=NaiveForecaster(strategy="last", sp=1),
        estimator_id="NaiveForecaster-last",
    )

    for idx, dataset_loader in enumerate(dataset_loaders):
        benchmark.add_task(
            dataset_loader,
            cv_splitter,
            scorers,
            task_id=f"patient_{idx}",
        )
    return benchmark


def run_benchmark():
    dataset_loaders = get_dataset_loaders()[:2]

    cv_splitter = get_cv_splitter()

    scorers = [PinballLoss(), MeanSquaredError(square_root=True)]

    benchmark = get_benchmark(dataset_loaders, cv_splitter, scorers)

    return parse_output(benchmark)


if __name__ == "__main__":
    start_time = time.time()
    run_benchmark()
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
