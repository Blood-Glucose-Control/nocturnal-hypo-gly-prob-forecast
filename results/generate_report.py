import pandas as pd
import os
import re

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_RESULTS_PATH = PROJECT_ROOT_DIR + "/results/processed"


class ModelRun:
    def __init__(self, filename, date, time, run, model_spec, interval):
        self.pnum = None
        self.interval = interval
        self.model = None
        self.model_id = None
        self.model_spec = model_spec
        self.date = date
        self.mse_mean = None
        self.mse_std = None
        self.runtime_secs = None
        self.filename = filename
        self.run = run
        self.time = time

    def copy_with_results(self, pnum, model, model_id, runtime_secs, mse_mean, mse_std):
        new_run = ModelRun(
            filename=self.filename,
            date=self.date,
            time=self.time,
            run=self.run,
            model_spec=self.model_spec,
            interval=self.interval,
        )
        new_run.pnum = pnum
        new_run.model = model
        new_run.model_id = model_id
        new_run.runtime_secs = runtime_secs
        new_run.mse_mean = mse_mean
        new_run.mse_std = mse_std
        return new_run

    def __str__(self):
        return f"ModelRun(filename={self.filename}, date={self.date}, time={self.time}, run={self.run}, model_spec={self.model_spec}, interval={self.interval}, pnum={self.pnum}, model={self.model}, model_id={self.model_id}, runtime_secs={self.runtime_secs}, mse_mean={self.mse_mean}, mse_std={self.mse_std})"


def create_model_run_from_filename(filename):
    pattern = r"""
    (\d{4}-\d{2}-\d{2})     # Date
    _
    (\d{2}-\d{2}-\d{2})     # Time
    _
    (\d+)                   # Run number
    _
    (.+?)                   # Model specification (everything until the time interval)
    _
    (\d+min)                # Time interval
    """

    match = re.match(pattern, filename, re.VERBOSE)
    if match:
        return ModelRun(
            filename=filename,
            date=match.group(1),
            time=match.group(2),
            run=match.group(3),
            model_spec=match.group(4),
            interval=match.group(5),
        )
    return None

def save_all_model_runs_to_csv(model_runs: list[ModelRun], filename: str = "report.csv"):
    df = pd.DataFrame([model_run.__dict__ for model_run in model_runs])
    # Sort by pnum so rows with same patient number are grouped together
    df = df.sort_values(by='pnum')
    df.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/{filename}", index=False)

def get_all_csv_files():
    all_csv_files = []
    for root, _, files in os.walk(PROCESSED_RESULTS_PATH):
        csv_files = [f for f in files if f.endswith(".csv")]
        if root.split("/")[-1] == "old" or len(csv_files) == 0:
            continue
        all_csv_files.extend([os.path.join(root, f) for f in csv_files])
    return all_csv_files


def get_best_model_idx(group):
    valid_mse = group[group["MeanSquaredError_mean"].notna()]
    if not valid_mse.empty:
        return valid_mse["MeanSquaredError_mean"].idxmin()
    # Fallback to runtime if all MSE values are NaN
    return group["runtime_secs"].idxmin()


def main():
    all_csv_files = get_all_csv_files()
    all_model_runs = []

    for file in all_csv_files:
        filename = file.split("/")[-1]
        model_run = create_model_run_from_filename(filename)

        if model_run is None:
            continue

        df = pd.read_csv(file)
        df["model"] = df["model_id"].apply(lambda x: x.split("-")[0])

        # Group by both model type and patient
        best_model_indices = df.groupby(["model", "validation_id"]).apply(
            get_best_model_idx
        )
        best_models = df.loc[best_model_indices]
        for _, row in best_models.iterrows():
            all_model_runs.append(
                model_run.copy_with_results(
                    row["validation_id"],
                    row["model"],
                    row["model_id"],
                    row["runtime_secs"],
                    row["MeanSquaredError_mean"],
                    row["MeanSquaredError_std"],
                )
            )

    save_all_model_runs_to_csv(all_model_runs)


if __name__ == "__main__":
    main()
