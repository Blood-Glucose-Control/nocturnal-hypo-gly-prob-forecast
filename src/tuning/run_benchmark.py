import time
from benchmark import run_benchmark

if __name__ == "__main__":
    start_time = time.time()
    print("Starting..... ")

    run_benchmark(
        y_features=["bg-0:00"],
        x_features=["iob", "cob"],
        steps_per_hour=12,
        hours_to_forecast=6,
        yaml_path="./src/tuning/configs/old/modset1.yaml",
        bg_method="linear",
        hr_method="linear",
        step_method="constant",
        cal_method="constant",
        processed_dir="./results/processed",
        raw_dir="./results/raw",
        cores_num=-1,
        n_patients=1,
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
