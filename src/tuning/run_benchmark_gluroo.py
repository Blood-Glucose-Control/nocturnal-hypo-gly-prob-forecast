import time
from benchmark import run_benchmark


# Change this
# yaml_path = "./src/tuning/configs/0_naive_05min.yaml"
# yaml_path = "./src/tuning/configs/1_arma_05min.yaml"
# yaml_path = "./src/tuning/configs/3_exponential_smooth_05min.yaml"
yaml_path = "./src/tuning/configs/8_ensemble_gk_05min.yaml"
if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting..... {yaml_path.split('/')[-1]}")

    run_benchmark(
        data_source_name="gluroo",
        y_features=["bg-0:00"],
        x_features=[
            # "hr-0:00", // DNE
            # "steps-0:00", // DNE
            # "cals-0:00", // DNE
            "cob",  # From carbs-0:00
            "carb_availability",  # From carbs-0:00
            "insulin_availability",  # From insulin-0:00
            "iob",  # From insulin-0:00
        ],
        initial_cv_window=12 * 24 * 3,  # 3 days
        cv_step_length=12 * 24 * 6,  # 6 days
        steps_per_hour=12,
        hours_to_forecast=6,
        yaml_path=yaml_path,
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        results_dir="./results",
        cores_num=-1,  # All cores
        n_patients=-1,  # All patients
        is_5min=True,  # 5-minute interval patients
    )
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
