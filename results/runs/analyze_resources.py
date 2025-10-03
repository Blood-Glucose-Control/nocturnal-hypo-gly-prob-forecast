#!/usr/bin/env python3
"""
Script to analyze SLURM resource usage and performance metrics across training runs.
Helps identify optimal resource configurations for L40S GPU training.
"""

import re
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Optional


class RunAnalyzer:
    def __init__(self, runs_dir="ttm_finetune"):
        self.runs_dir = Path(__file__).parent / runs_dir
        self.runs_data = []

    def parse_run_info(self, run_dir: Path) -> Optional[Dict]:
        """Parse run_info.txt file to extract resource and timing data."""
        info_file = run_dir / "run_info.txt"
        if not info_file.exists():
            return None

        run_data = {
            "run_directory": run_dir.name,
            "job_id": None,
            "node": None,
            "start_time": None,
            "end_time": None,
            "elapsed_seconds": None,
            "training_exit_code": None,
            "slurm_settings": {},
            "gpu_info": {},
            "performance_metrics": {},
        }

        try:
            with open(info_file, "r") as f:
                content = f.read()

            # Parse basic run information
            if match := re.search(r"Job ID: (\d+)", content):
                run_data["job_id"] = match.group(1)

            if match := re.search(r"Node: (\S+)", content):
                run_data["node"] = match.group(1)

            if match := re.search(r"Start Time: (.+)", content):
                run_data["start_time"] = match.group(1)

            if match := re.search(r"End Time: (.+)", content):
                run_data["end_time"] = match.group(1)

            if match := re.search(r"Total Elapsed Seconds: (\d+)", content):
                run_data["elapsed_seconds"] = int(match.group(1))

            if match := re.search(r"Training Exit Code: (\d+)", content):
                run_data["training_exit_code"] = int(match.group(1))

            # Parse SLURM resource settings
            slurm_patterns = {
                "cpus_per_task": r"CPUs per Task: (\d+)",
                "memory_per_node": r"Memory per Node: (\d+)",
                "memory_per_cpu": r"Memory per CPU: (\d+)",
                "partition": r"Partition: (\S+)",
                "time_limit": r"Time Limit: (.+)",
                "qos": r"QOS: (\S+)",
                "num_nodes": r"Number of Nodes: (\d+)",
                "num_tasks": r"Number of Tasks: (\d+)",
                "gpus_per_node": r"GPUs per Node: (\d+)",
                "gpu_type": r"GPU Type: (.+)",
                "account": r"Account: (\S+)",
            }

            for key, pattern in slurm_patterns.items():
                if match := re.search(pattern, content):
                    value = match.group(1)
                    # Convert numeric values
                    if key in [
                        "cpus_per_task",
                        "memory_per_node",
                        "memory_per_cpu",
                        "num_nodes",
                        "num_tasks",
                        "gpus_per_node",
                    ]:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    run_data["slurm_settings"][key] = value

            # Parse GPU information if available
            if "GPU Name:" in content:
                if match := re.search(r"GPU Name: (.+)", content):
                    run_data["gpu_info"]["name"] = match.group(1).strip()

                if match := re.search(r"GPU Memory: (\d+)", content):
                    run_data["gpu_info"]["memory_mb"] = int(match.group(1))

                if match := re.search(r"GPU Driver Version: (.+)", content):
                    run_data["gpu_info"]["driver_version"] = match.group(1).strip()

            return run_data

        except Exception as e:
            print(f"Error parsing {info_file}: {e}")
            return None

    def parse_training_log(self, run_dir: Path, run_data: Dict) -> Dict:
        """Parse training.log to extract performance metrics."""
        training_log = run_dir / "training.log"
        if not training_log.exists():
            return run_data

        try:
            with open(training_log, "r") as f:
                content = f.read()

            # Extract final training metrics
            if match := re.search(r"'train_samples_per_second': ([\d.]+)", content):
                run_data["performance_metrics"]["train_samples_per_second"] = float(
                    match.group(1)
                )

            if match := re.search(r"'train_steps_per_second': ([\d.]+)", content):
                run_data["performance_metrics"]["train_steps_per_second"] = float(
                    match.group(1)
                )

            if match := re.search(r"'train_loss': ([\d.]+)", content):
                run_data["performance_metrics"]["final_train_loss"] = float(
                    match.group(1)
                )

            if match := re.search(r"'eval_loss': ([\d.]+)", content):
                run_data["performance_metrics"]["final_eval_loss"] = float(
                    match.group(1)
                )

            # Extract batch size information
            if match := re.search(r"'custom_batch_size': (\d+)", content):
                run_data["performance_metrics"]["batch_size"] = int(match.group(1))

            if match := re.search(r"'custom_learning_rate': ([\d.]+)", content):
                run_data["performance_metrics"]["learning_rate"] = float(match.group(1))

            # Count total training steps
            train_steps = len(re.findall(r"'train_steps_per_second':", content))
            if train_steps > 0:
                run_data["performance_metrics"]["total_training_steps"] = train_steps

        except Exception as e:
            print(f"Error parsing training log for {run_dir}: {e}")

        return run_data

    def parse_gpu_utilization(self, run_dir: Path, run_data: Dict) -> Dict:
        """Parse GPU utilization logs to get average utilization metrics."""
        util_log = run_dir / "gpu_utilization.log"
        if not util_log.exists():
            return run_data

        try:
            with open(util_log, "r") as f:
                lines = f.readlines()

            gpu_utils = []
            memory_utils = []
            temperatures = []
            power_draws = []

            for line in lines:
                # Format: "YYYY-MM-DD HH:MM:SS: gpu_util, memory_util, temp, power"
                if match := re.search(
                    r": ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)", line
                ):
                    gpu_utils.append(float(match.group(1)))
                    memory_utils.append(float(match.group(2)))
                    temperatures.append(float(match.group(3)))
                    power_draws.append(float(match.group(4)))

            if gpu_utils:
                run_data["performance_metrics"]["avg_gpu_utilization"] = sum(
                    gpu_utils
                ) / len(gpu_utils)
                run_data["performance_metrics"]["max_gpu_utilization"] = max(gpu_utils)
                run_data["performance_metrics"]["min_gpu_utilization"] = min(gpu_utils)

            if memory_utils:
                run_data["performance_metrics"]["avg_memory_utilization"] = sum(
                    memory_utils
                ) / len(memory_utils)
                run_data["performance_metrics"]["max_memory_utilization"] = max(
                    memory_utils
                )

            if temperatures:
                run_data["performance_metrics"]["avg_temperature"] = sum(
                    temperatures
                ) / len(temperatures)
                run_data["performance_metrics"]["max_temperature"] = max(temperatures)

            if power_draws:
                run_data["performance_metrics"]["avg_power_draw"] = sum(
                    power_draws
                ) / len(power_draws)
                run_data["performance_metrics"]["max_power_draw"] = max(power_draws)

        except Exception as e:
            print(f"Error parsing GPU utilization for {run_dir}: {e}")

        return run_data

    def analyze_runs(self):
        """Analyze all runs in the directory."""
        if not self.runs_dir.exists():
            print(f"Runs directory {self.runs_dir} not found")
            return

        run_dirs = [
            d
            for d in self.runs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        run_dirs.sort()

        print(f"Found {len(run_dirs)} run directories")
        print("=" * 80)

        for run_dir in run_dirs:
            run_data = self.parse_run_info(run_dir)
            if run_data:
                run_data = self.parse_training_log(run_dir, run_data)
                run_data = self.parse_gpu_utilization(run_dir, run_data)
                self.runs_data.append(run_data)

        if not self.runs_data:
            print("No valid run data found")
            return

        self.print_summary()
        self.print_resource_performance_analysis()
        self.save_analysis_json()

    def print_summary(self):
        """Print a summary of all runs."""
        print(f"\n=== Summary of {len(self.runs_data)} Runs ===")
        print(
            f"{'Job ID':<8} {'Node':<12} {'CPUs':<5} {'Mem(GB)':<8} {'Duration':<10} {'Steps/s':<8} {'GPU Util':<9} {'Status':<8}"
        )
        print("-" * 80)

        for run in self.runs_data:
            job_id = run.get("job_id", "N/A")
            node = run.get("node", "N/A")
            cpus = run["slurm_settings"].get("cpus_per_task", "N/A")

            mem_mb = run["slurm_settings"].get("memory_per_node", 0)
            mem_gb = f"{mem_mb/1024:.1f}" if mem_mb else "N/A"

            elapsed = run.get("elapsed_seconds", 0)
            duration = f"{elapsed//3600}h{(elapsed%3600)//60}m" if elapsed else "N/A"

            steps_per_sec = run["performance_metrics"].get(
                "train_steps_per_second", "N/A"
            )
            if isinstance(steps_per_sec, float):
                steps_per_sec = f"{steps_per_sec:.2f}"

            gpu_util = run["performance_metrics"].get("avg_gpu_utilization")
            gpu_util_str = f"{gpu_util:.1f}%" if gpu_util else "N/A"

            status = "Success" if run.get("training_exit_code") == 0 else "Failed"

            print(
                f"{job_id:<8} {node:<12} {cpus:<5} {mem_gb:<8} {duration:<10} {steps_per_sec:<8} {gpu_util_str:<9} {status:<8}"
            )

    def print_resource_performance_analysis(self):
        """Analyze resource allocation vs performance."""
        print("\n=== Resource vs Performance Analysis ===")

        # Group by CPU count
        cpu_groups = {}
        for run in self.runs_data:
            cpus = run["slurm_settings"].get("cpus_per_task", 0)
            if cpus not in cpu_groups:
                cpu_groups[cpus] = []
            cpu_groups[cpus].append(run)

        print("\n--- Performance by CPU Count ---")
        for cpus, runs in sorted(cpu_groups.items()):
            if not runs:
                continue

            avg_steps = [
                r["performance_metrics"].get("train_steps_per_second", 0)
                for r in runs
                if r["performance_metrics"].get("train_steps_per_second")
            ]
            avg_gpu_util = [
                r["performance_metrics"].get("avg_gpu_utilization", 0)
                for r in runs
                if r["performance_metrics"].get("avg_gpu_utilization")
            ]

            if avg_steps:
                avg_steps_val = sum(avg_steps) / len(avg_steps)
                avg_gpu_val = (
                    sum(avg_gpu_util) / len(avg_gpu_util) if avg_gpu_util else 0
                )
                print(
                    f"  {cpus} CPUs: {len(runs)} runs, avg {avg_steps_val:.2f} steps/s, avg {avg_gpu_val:.1f}% GPU util"
                )

        # Memory analysis
        print("\n--- Performance by Memory Allocation ---")
        mem_groups = {}
        for run in self.runs_data:
            mem_gb = run["slurm_settings"].get("memory_per_node", 0) / 1024
            mem_bucket = int(mem_gb // 8) * 8  # Group by 8GB buckets
            if mem_bucket not in mem_groups:
                mem_groups[mem_bucket] = []
            mem_groups[mem_bucket].append(run)

        for mem_gb, runs in sorted(mem_groups.items()):
            if not runs:
                continue

            avg_steps = [
                r["performance_metrics"].get("train_steps_per_second", 0)
                for r in runs
                if r["performance_metrics"].get("train_steps_per_second")
            ]
            if avg_steps:
                avg_steps_val = sum(avg_steps) / len(avg_steps)
                print(
                    f"  {mem_gb}GB+ Memory: {len(runs)} runs, avg {avg_steps_val:.2f} steps/s"
                )

    def save_analysis_json(self):
        """Save detailed analysis to JSON file."""
        output_file = self.runs_dir.parent / "run_analysis.json"

        analysis_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_runs": len(self.runs_data),
            "runs": self.runs_data,
        }

        with open(output_file, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        print(f"\nDetailed analysis saved to: {output_file}")


def main():
    analyzer = RunAnalyzer()
    analyzer.analyze_runs()


if __name__ == "__main__":
    main()
