# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

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
        self.pinball_mean = None
        self.pinball_std = None
        self.runtime_secs = None
        self.filename = filename
        self.run = run
        self.time = time

    def copy_with_results(
        self,
        pnum,
        model,
        model_id,
        runtime_secs,
        mse_mean,
        mse_std,
        pinball_mean,
        pinball_std,
    ):
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
        new_run.pinball_mean = pinball_mean
        new_run.pinball_std = pinball_std
        return new_run

    def __str__(self):
        return f"ModelRun(filename={self.filename}, date={self.date}, time={self.time}, run={self.run}, model_spec={self.model_spec}, interval={self.interval}, pnum={self.pnum}, model={self.model}, model_id={self.model_id}, runtime_secs={self.runtime_secs}, mse_mean={self.mse_mean}, mse_std={self.mse_std})"


def create_model_run_from_filename(filename):
    # Try first with run number
    pattern = r"""
    (\d{4}-\d{2}-\d{2})      # Date
    _
    (\d{2}-\d{2}-\d{2})      # Time
    _
    (\d+)                    # Run number
    _
    (.+?)                    # Model specification
    _
    (\d+min)                 # Time interval
    """

    match = re.match(pattern, filename, re.VERBOSE)
    if match:
        return ModelRun(
            filename=filename,
            date=pd.to_datetime(match.group(1)),
            time=match.group(2),
            run=match.group(3),
            model_spec=match.group(4),
            interval=match.group(5),
        )

    # If first pattern fails, try without run number
    pattern = r"""
    (\d{4}-\d{2}-\d{2})     # Date
    _
    (\d{2}-\d{2}-\d{2})     # Time
    _
    (.+?)                    # Model specification
    _
    (\d+min)                 # Time interval
    """

    match = re.match(pattern, filename, re.VERBOSE)
    if match:
        return ModelRun(
            filename=filename,
            date=pd.to_datetime(match.group(1)),
            time=match.group(2),
            run="0",  # Default run number
            model_spec=match.group(3),
            interval=match.group(4),
        )

    return None


def save_all_model_runs_to_csv(
    model_runs: list[ModelRun], filename: str = "report.csv"
):
    df = pd.DataFrame([model_run.__dict__ for model_run in model_runs])
    # Sort by pnum so rows with same patient number are grouped together
    df = df.sort_values(by="pnum")
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
    # check if pinball loss is available
    if "PinballLoss_mean" in group.columns:
        valid_pinball = group[group["PinballLoss_mean"].notna()]
        if not valid_pinball.empty:
            return valid_pinball["PinballLoss_mean"].idxmin()
    valid_mse = group[group["MeanSquaredError_mean"].notna()]
    if not valid_mse.empty:
        return valid_mse["MeanSquaredError_mean"].idxmin()
    # Fallback to runtime if all MSE values are NaN
    return group["runtime_secs"].idxmin()


def main():
    all_csv_files = get_all_csv_files()
    all_model_runs: list[ModelRun] = []

    for file in all_csv_files:
        filename = file.split("/")[-1]
        model_run = create_model_run_from_filename(filename)

        if model_run is None:
            continue

        df = pd.read_csv(file)
        df["model"] = df["model_id"].apply(lambda x: x.split("-")[0])

        for patient, patient_df in df.groupby("validation_id"):
            for model, model_df in patient_df.groupby("model"):
                if (
                    "PinballLoss_mean" in model_df.columns
                    and not model_df["PinballLoss_mean"].isna().all()
                ):
                    best_model = model_df.sort_values("PinballLoss_mean").iloc[0]
                elif not model_df["MeanSquaredError_mean"].isna().all():
                    best_model = model_df.sort_values("MeanSquaredError_mean").iloc[0]
                else:
                    best_model = model_df.sort_values("runtime_secs").iloc[0]

                all_model_runs.append(
                    model_run.copy_with_results(
                        pnum=patient,
                        model=model,
                        model_id=best_model["model_id"],
                        runtime_secs=best_model["runtime_secs"],
                        mse_mean=best_model["MeanSquaredError_mean"],
                        mse_std=best_model["MeanSquaredError_std"],
                        pinball_mean=(
                            best_model["PinballLoss_mean"]
                            if "PinballLoss_mean" in best_model
                            else None
                        ),
                        pinball_std=(
                            best_model["PinballLoss_std"]
                            if "PinballLoss_std" in best_model
                            else None
                        ),
                    )
                )
    deduped_model_runs_dict = {}
    seen_keys = set()
    for model_run in all_model_runs:
        key = (model_run.pnum, model_run.model_id)
        if key in seen_keys:
            current_best = deduped_model_runs_dict[key]
            if model_run.date >= current_best.date:
                deduped_model_runs_dict[key] = model_run
        else:
            seen_keys.add(key)
            deduped_model_runs_dict[key] = model_run

    deduped_model_runs = list(deduped_model_runs_dict.values())

    save_all_model_runs_to_csv(deduped_model_runs)


if __name__ == "__main__":
    main()
