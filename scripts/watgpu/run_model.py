import time
from src.tuning.benchmark import run_benchmark
import os
import sys

# Get yaml filename from command line argument
if len(sys.argv) != 4:
    print("Usage: python run_benchmark.py <yaml_filename> <description> <timestamp>")
    sys.exit(1)

yaml_filename = sys.argv[1]
description = sys.argv[2]
timestamp = sys.argv[3]

# There is definitely a better way to do this
results_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results"
)
yaml_path = os.path.join(
    os.environ["HOME"],
    f"nocturnal-hypo-gly-prob-forecast/src/tuning/configs/{yaml_filename}",
)

# Set the configuration based on filename
is_5min = "05min" in yaml_filename
config = {
    "steps_per_hour": 12 if is_5min else 4,
    "is_5min": is_5min,
}

print("config: ", config)

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting..... {yaml_filename}")

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
        results_dir=results_dir,
        cores_num=-1,  # Use all cores
        n_patients=-1,  # Use all patients for the given interval
        is_5min=config["is_5min"],
        description=description,
        timestamp=timestamp,
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
