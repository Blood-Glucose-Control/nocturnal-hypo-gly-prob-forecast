import time
from src.tuning.benchmark import run_benchmark
import os

processed_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results/processed"
)
raw_dir = os.path.join(
    os.environ["HOME"], "nocturnal-hypo-gly-prob-forecast/results/raw"
)
yaml_path = os.path.join(
    os.environ["HOME"],
    "nocturnal-hypo-gly-prob-forecast/src/tuning/configs/0_arma_15min.yaml",
)

if __name__ == "__main__":
    start_time = time.time()
    print("Starting..... 0_arma_15min.yaml")

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
        yaml_path=yaml_path,
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        cores_num=-1,  # Use all cores
        n_patients=-1,  # Use all patients
        is_5min=False,  # 15-min interval
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
