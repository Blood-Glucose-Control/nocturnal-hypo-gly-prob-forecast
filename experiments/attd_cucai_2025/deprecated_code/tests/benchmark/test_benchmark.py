# import json
# import os
# import time
# from datetime import datetime

# import pytest

# from src.tuning.benchmark import run_benchmark


# class TestBenchmark:
#     timestamp = "2025-03-20_18-54-56"
#     yaml_path = "./src/tuning/configs/0_naive_05min.yaml"
#     config = {
#         "steps_per_hour": 12,
#         "is_5min": True,
#     }
#     yaml_name = yaml_path.split("/")[-1].replace(".yaml", "")
#     results_dir = "./results"
#     raw_output_file_path = (
#         "./results/raw/tests/"
#         + datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
#         + "_0_naive_05min.csv"
#     )
#     processed_dir = "./results/processed/tests/" + datetime.now().strftime(
#         "%Y-%m-%dT%H:%M:%S"
#     )
#     x_features = [
#         "hr_bpm",
#         "steps",
#         "cals",
#         "cob",
#         "carb_availability",
#         "insulin_availability",
#         "iob",
#     ]
#     y_features = ["bg_mM"]

#     @pytest.fixture(scope="class")
#     def benchmark_outputs(self):
#         # Run benchmark once
#         print("Running benchmark - benchmark_outputs fixture")
#         run_benchmark(
#             data_source_name="kaggle_brisT1D",
#             y_features=TestBenchmark.y_features,
#             x_features=TestBenchmark.x_features,
#             initial_cv_window=TestBenchmark.config["steps_per_hour"] * 24 * 3,  # 3 days
#             cv_step_length=TestBenchmark.config["steps_per_hour"] * 24 * 6,  # 6 days
#             steps_per_hour=TestBenchmark.config["steps_per_hour"],
#             hours_to_forecast=6,
#             yaml_path=TestBenchmark.yaml_path,
#             bg_method="linear",
#             hr_method="linear",
#             step_method="constant",
#             cal_method="constant",
#             results_dir=TestBenchmark.results_dir,
#             cores_num=1,  # 1 core is sufficient for testing
#             n_patients=2,  # 2 patients are sufficient for testing
#             is_5min=TestBenchmark.config["is_5min"],  # 5-minute interval patients
#             description="Test benchmark",
#             timestamp=TestBenchmark.timestamp,
#         )

#     def test_raw_output_file_exists(self, benchmark_outputs):
#         """Test that the benchmark creates the raw output file"""
#         assert os.path.exists(self.raw_output_file_path)

#     def test_processed_dir_exists(self, benchmark_outputs):
#         """Test that the benchmark creates the expected output directories"""
#         time.sleep(3)
#         assert os.path.exists(self.processed_dir)
#         assert os.path.exists(
#             self.processed_dir + "/meansquarederror/naive/05mins/naive.csv"
#         )
#         assert os.path.exists(
#             self.processed_dir + "/pinballloss/naive/05mins/naive.csv"
#         )

#     def test_run_configs_exists(self, benchmark_outputs):
#         """Test that the benchmark creates the expected run config file"""
#         assert os.path.exists(
#             self.processed_dir + "/configs/0_naive_05min_run_config.json"
#         )

#     def test_run_config_content(self, benchmark_outputs):
#         """Test that the run config file contains the expected content"""
#         config_path = os.path.join(
#             self.processed_dir, "configs", "0_naive_05min_run_config.json"
#         )
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # Verify key configuration parameters match what was passed to run_benchmark
#         assert config["data_source_name"] == "kaggle_brisT1D"
#         assert config["y_features"] == TestBenchmark.y_features
#         assert config["x_features"] == TestBenchmark.x_features
#         assert config["hours_to_forecast"] == 6
#         assert config["steps_per_hour"] == 12
#         assert len(config["patient_numbers"]) == 2
#         assert config["description"] == "Test benchmark"

#     # def test_cleanup(self, benchmark_outputs):
#     #     """Test that cleanup removes all generated files and directories"""
#     #     shutil.rmtree(self.processed_dir)
#     #     os.remove(self.raw_output_file_path)
#     #
#     #     assert not os.path.exists(self.raw_output_file_path)
#     #     assert not os.path.exists(self.processed_dir)
