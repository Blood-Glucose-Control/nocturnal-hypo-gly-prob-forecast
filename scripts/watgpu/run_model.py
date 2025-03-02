import time
from src.tuning.benchmark import run_benchmark
import os
import sys

# Get yaml filename from command line argument
if len(sys.argv) != 2:
    print("Usage: python run_benchmark.py <yaml_filename>")
    sys.exit(1)

yaml_filename = sys.argv[1]

processed_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results/processed"
)
raw_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results/raw"
)
config_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results/configs"
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

    try:
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
            processed_dir=processed_dir,
            raw_dir=raw_dir,
            config_dir=config_dir,
            cores_num=-1,  # All cores
            n_patients=-1,  # All patients
            is_5min=config["is_5min"],
        )

        end_time = time.time()
        print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.time()
        print(f"Error: {e}")
        print(f"Benchmark failed after {end_time - start_time:.2f} seconds")
