import time
from benchmark import run_benchmark


# Change this
yaml_path = "./src/tuning/configs/0_naive_05min.yaml"


is_5min = "05min" in yaml_path  # Will be False since this is 15min
config = {}
if is_5min:
    config = {
        "steps_per_hour": 12,
        "is_5min": True,
    }
else:
    config = {
        "steps_per_hour": 4,
        "is_5min": False,
    }


if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting..... {yaml_path.split('/')[-1]}")

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
        initial_cv_window=config["steps_per_hour"] * 24 * 3,  # 3 days
        cv_step_length=config["steps_per_hour"] * 24 * 6,  # 6 days
        steps_per_hour=config["steps_per_hour"],
        hours_to_forecast=6,
        yaml_path=yaml_path,
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        results_dir="./results",
        cores_num=-1,  # All cores
        n_patients=-1,  # All patients
        is_5min=config["is_5min"],  # 5-minute interval patients
    )
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
