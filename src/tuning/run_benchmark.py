import time
from benchmark import run_benchmark

if __name__ == "__main__":
    start_time = time.time()
    print("Starting..... ")

    run_benchmark(
        y_features=["bg-0:00"],
        x_features=[
            "hr-0:00",
            "steps-0:00",
            "cals-0:00",
            "cob",
            "carb_availability",
            "insulin_availability",
            "iob",
        ],
        initial_cv_window=4 * 24 * 3,  # 3 days
        cv_step_length=4 * 24 * 3,  # 3 days
        steps_per_hour=4,
        hours_to_forecast=6,
        yaml_path="./src/tuning/configs/1_exponential_smooth_15min.yaml",
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        processed_dir="./results/processed",
        raw_dir="./results/raw",
        cores_num=-1,  # All cores
        n_patients=-1,  # All patients
        is_5min=False,  # 5-minute interval patients
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
