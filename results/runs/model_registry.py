#!/usr/bin/env python3
"""
Model Registry System for TTM Training Runs
Maintains a CSV database of all training experiments with their configurations and results.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


class ModelRegistry:
    def __init__(self, registry_path="results/runs/ttm_finetune/model_registry.csv"):
        """Initialize model registry with CSV file path"""
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_registry()

    def _initialize_registry(self):
        """Create registry CSV file with headers if it doesn't exist"""
        if not self.registry_path.exists():
            headers = [
                # Run Metadata
                "run_id",
                "job_id",
                "experiment_name",
                "description",
                "tags",
                "version",
                "start_time",
                "end_time",
                "elapsed_time_seconds",
                "status",
                # Hardware Information
                "node_name",
                "gpu_type",
                "gpu_memory_gb",
                "cpu_count",
                "total_memory_gb",
                "cpus_per_task",
                "mem_per_cpu_gb",
                "partition",
                "time_limit",
                # Model Configuration
                "model_path",
                "context_length",
                "forecast_length",
                # Data Configuration
                "data_source",
                "y_feature",
                "x_features",
                "timestamp_column",
                "resolution_min",
                "data_split_train",
                "data_split_test",
                "fewshot_percent",
                # Training Configuration
                "batch_size",
                "learning_rate",
                "num_epochs",
                "loss_function",
                "optimizer",
                "scheduler",
                "mixed_precision",
                "resume_dir",
                "dataloader_num_workers",
                # Results
                "final_train_loss",
                "final_eval_loss",
                "best_eval_loss",
                "best_checkpoint",
                "training_samples_per_second",
                "peak_gpu_utilization",
                "peak_memory_usage",
                # File Paths
                "run_directory",
                "config_file",
                "model_output_dir",
                "log_files",
            ]

            with open(self.registry_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def register_run_start(
        self, run_dir, config=None, slurm_info=None, hardware_info=None
    ):
        """Register the start of a new training run"""
        run_info = self._extract_run_info(run_dir, config, slurm_info, hardware_info)
        run_info["status"] = "running"
        run_info["start_time"] = datetime.now().isoformat()

        # Read existing data
        df = (
            pd.read_csv(self.registry_path)
            if self.registry_path.exists()
            else pd.DataFrame()
        )

        # Add new run
        new_run_df = pd.DataFrame([run_info])
        df = pd.concat([df, new_run_df], ignore_index=True)

        # Save back to CSV
        df.to_csv(self.registry_path, index=False)

        return run_info["run_id"]

    def register_run_completion(self, run_id, results=None, status="completed"):
        """Update registry with completion information and extract metrics"""
        df = pd.read_csv(self.registry_path)

        mask = df["run_id"] == run_id
        if not mask.any():
            print(f"Warning: Run ID {run_id} not found in registry")
            return

        # Update basic completion info
        df.loc[mask, "status"] = status
        df.loc[mask, "end_time"] = datetime.now().isoformat()

        # Get run directory and extract GPU metrics
        run_dir = df.loc[mask, "run_directory"].iloc[0]

        # Extract GPU utilization from monitoring log
        gpu_log_path = f"{run_dir}/gpu_utilization.log"
        if os.path.exists(gpu_log_path):
            try:
                # Skip header lines and read CSV
                gpu_df = pd.read_csv(
                    gpu_log_path, skiprows=3
                )  # Skip the 3 header lines
                if not gpu_df.empty and len(gpu_df.columns) >= 3:
                    df.loc[mask, "peak_gpu_utilization"] = gpu_df.iloc[
                        :, 1
                    ].max()  # GPU_Util_%
                    df.loc[mask, "peak_memory_usage"] = gpu_df.iloc[
                        :, 2
                    ].max()  # Memory_Util_%
            except Exception as e:
                print(f"Warning: Could not parse GPU utilization log: {e}")

        # Extract training metrics from training log
        training_log_path = f"{run_dir}/training.log"
        if os.path.exists(training_log_path):
            try:
                with open(training_log_path, "r") as f:
                    log_content = f.read()

                # Extract final losses (adapt regex to your log format)
                import re

                final_train_match = re.search(
                    r"Final train loss:\s*([\d\.]+)", log_content
                )
                if final_train_match:
                    df.loc[mask, "final_train_loss"] = float(final_train_match.group(1))

                final_eval_match = re.search(
                    r"Final eval loss:\s*([\d\.]+)", log_content
                )
                if final_eval_match:
                    df.loc[mask, "final_eval_loss"] = float(final_eval_match.group(1))

                best_eval_match = re.search(r"Best eval loss:\s*([\d\.]+)", log_content)
                if best_eval_match:
                    df.loc[mask, "best_eval_loss"] = float(best_eval_match.group(1))

            except Exception as e:
                print(f"Warning: Could not parse training log: {e}")

        # Update with any additional results passed in
        if results:
            for key, value in results.items():
                if key in df.columns:
                    df.loc[mask, key] = value

        # Calculate elapsed time
        if df.loc[mask, "start_time"].notna().iloc[0]:
            start_time = pd.to_datetime(df.loc[mask, "start_time"].iloc[0])
            end_time = pd.to_datetime(df.loc[mask, "end_time"].iloc[0])
            elapsed_seconds = (end_time - start_time).total_seconds()
            df.loc[mask, "elapsed_time_seconds"] = elapsed_seconds

        df.to_csv(self.registry_path, index=False)

    def _extract_run_info(
        self, run_dir, config=None, slurm_info=None, hardware_info=None
    ):
        """Extract run information from various sources"""
        run_path = Path(run_dir)
        run_info = {}

        # Basic run information
        run_info["run_id"] = run_path.name
        run_info["run_directory"] = str(run_path)

        # Extract job ID from run directory name
        import re

        job_match = re.search(r"job(\d+)", run_path.name)
        run_info["job_id"] = job_match.group(1) if job_match else None

        # Configuration information
        if config:
            exp_config = config.get("experiment", {})
            model_config = config.get("model", {})
            data_config = config.get("data", {})
            training_config = config.get("training", {})
            hardware_config = config.get("hardware", {})
            output_config = config.get("output", {})
            optimization_config = config.get("optimization", {})
            evaluation_config = config.get("evaluation", {})

            # Experiment metadata
            run_info["experiment_name"] = exp_config.get("name", "")
            run_info["description"] = exp_config.get("description", "")
            run_info["tags"] = ",".join(exp_config.get("tags", []))
            run_info["version"] = exp_config.get("version", "")

            # Model configuration
            run_info["model_path"] = model_config.get("path", "")
            run_info["context_length"] = model_config.get("context_length", "")
            run_info["forecast_length"] = model_config.get("forecast_length", "")

            # Data configuration
            run_info["data_source"] = data_config.get("source_name", "")
            run_info["y_feature"] = ",".join(data_config.get("y_feature", []))
            run_info["x_features"] = ",".join(data_config.get("x_features", []))
            run_info["timestamp_column"] = data_config.get("timestamp_column", "")
            run_info["resolution_min"] = data_config.get("resolution_min", "")
            data_split = data_config.get("data_split", [])
            run_info["data_split_train"] = data_split[0] if len(data_split) > 0 else ""
            run_info["data_split_test"] = data_split[1] if len(data_split) > 1 else ""
            run_info["fewshot_percent"] = data_config.get("fewshot_percent", "")

            # Training configuration
            run_info["batch_size"] = training_config.get("batch_size", "")
            run_info["dataloader_num_workers"] = training_config.get(
                "dataloader_num_workers", ""
            )
            run_info["learning_rate"] = training_config.get("learning_rate", "")
            run_info["num_epochs"] = training_config.get("num_epochs", "")
            run_info["use_cpu"] = training_config.get("use_cpu", None)
            run_info["loss_function"] = training_config.get("loss", "")
            run_info["resume_dir"] = training_config.get("resume_dir", "Not Specified")
            run_info["mixed_precision"] = training_config.get("mixed_precision", None)

            # Expected Hardware configuration
            run_info["expected_gpu_type"] = hardware_config.get("expected_gpu_type", "")
            run_info["expected_memory_gb"] = hardware_config.get(
                "expected_memory_gb", ""
            )

            # Output configuration
            run_info["save_checkpoints"] = output_config.get("save_checkpoints", None)
            run_info["checkpoint_frequency"] = output_config.get(
                "checkpoint_frequency", None
            )
            run_info["save_best_model"] = output_config.get("save_best_model", None)
            run_info["log_frequency"] = output_config.get("log_frequency", None)
            run_info["model_output_dir"] = output_config.get("model_output_dir", "")

            # Optimization configuration
            run_info["scheduler"] = optimization_config.get("scheduler", "Default")
            run_info["scheduler_params"] = str(
                optimization_config.get("scheduler_params", {})
            )
            run_info["optimizer"] = optimization_config.get("optimizer", "Default")
            run_info["optimizer_params"] = str(
                optimization_config.get("optimizer_params", {})
            )

            # Evaluation configuration
            run_info["eval_frequency"] = evaluation_config.get("eval_frequency", None)
            run_info["eval_batch_size"] = evaluation_config.get("eval_batch_size", None)
            run_info["metrics"] = evaluation_config.get("metrics", [])

        # SLURM information
        if slurm_info:
            run_info["node_name"] = slurm_info.get("node_name", "")
            run_info["cpus_per_task"] = slurm_info.get("cpus_per_task", "")
            run_info["mem_per_cpu_gb"] = slurm_info.get("mem_per_cpu_gb", "")
            run_info["partition"] = slurm_info.get("partition", "")
            run_info["time_limit"] = slurm_info.get("time_limit", "")

        # Hardware information
        if hardware_info:
            run_info["cpu_count"] = hardware_info.get("cpu_count", "")
            run_info["total_memory_gb"] = hardware_info.get("total_memory_gb", "")
            run_info["gpu_type"] = hardware_info.get("gpu_type", "")
            run_info["gpu_memory_gb"] = hardware_info.get("gpu_memory_gb", "")

        # File paths
        run_info["config_file"] = (
            str(run_path / "experiment_config.yaml") if config else ""
        )
        run_info["log_files"] = f"{run_path}/training.log,{run_path}/gpu_monitoring.log"

        return run_info

    def get_summary(self):
        """Get summary statistics of all runs"""
        if not self.registry_path.exists():
            return "No runs registered yet."

        df = pd.read_csv(self.registry_path)

        summary = f"""
        Model Registry Summary
        ======================
        Total Runs: {len(df)}
        Completed Runs: {len(df[df["status"] == "completed"])}
        Running Runs: {len(df[df["status"] == "running"])}
        Failed Runs: {len(df[df["status"] == "failed"])}

        Recent Runs:
        {df.tail(5)[["run_id", "experiment_name", "status", "start_time"]].to_string(index=False)}
        """

        return summary

    def query_runs(self, **filters):
        """Query runs based on filters"""
        if not self.registry_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.registry_path)

        for column, value in filters.items():
            if column in df.columns:
                df = df[df[column] == value]

        return df


def create_registry_entry_from_run_info(run_info_file):
    """Helper function to create registry entry from run_info.txt file"""
    registry = ModelRegistry()

    # Parse run_info.txt file
    # run_info = {}
    slurm_info = {}
    hardware_info = {}

    # with open(run_info_file, "r") as f:
    # content = f.read()

    # Extract information using regex or simple parsing
    # This would need to be implemented based on the actual format of run_info.txt

    # Register the run
    run_id = registry.register_run_start(
        run_dir=Path(run_info_file).parent,
        slurm_info=slurm_info,
        hardware_info=hardware_info,
    )

    return run_id


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    print(registry.get_summary())
