import time
from benchmark import run_benchmark

if __name__ == "__main__":
    start_time = time.time()
    print("Starting..... ")

    run_benchmark(
        y_features=["bg-0:00"],
        x_features=["iob", "cob"],
        initial_cv_window=12 * 24 * 3,  # 3 days
        cv_step_length=12 * 24 * 3,  # 3 days
        steps_per_hour=12,
        hours_to_forecast=6,
        yaml_path="./src/tuning/configs/0_arma_05min.yaml",
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        processed_dir="./results/processed",
        raw_dir="./results/raw",
        cores_num=-1,  # All cores
        n_patients=-1,  # All patients
        is_5min=True,  # 5-minute interval patients
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
