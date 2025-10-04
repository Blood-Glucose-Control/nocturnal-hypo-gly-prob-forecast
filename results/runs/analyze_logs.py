#!/usr/bin/env python3
"""
Script to analyze SLURM job logs and extract GPU performance information.
This helps assess L40S GPU utilization and identify performance bottlenecks.
"""

import re
from datetime import datetime
from pathlib import Path


def analyze_job_logs(log_dir="slurm_logs"):
    """Analyze SLURM job logs to extract performance metrics."""

    log_path = Path(__file__).parent / log_dir
    if not log_path.exists():
        print(f"Log directory {log_path} not found")
        return

    job_files = list(log_path.glob("JOB*.out"))
    error_files = list(log_path.glob("JOB*-err.out"))

    print(f"Found {len(job_files)} job output files and {len(error_files)} error files")
    print("-" * 80)

    for job_file in sorted(job_files):
        job_match = re.search(r"JOB(\d+)\.out", job_file.name)
        if not job_match:
            continue
        job_id = job_match.group(1)
        error_file = log_path / f"JOB{job_id}-err.out"

        print(f"\n=== Job ID: {job_id} ===")

        # Get file timestamps
        job_mtime = datetime.fromtimestamp(job_file.stat().st_mtime)
        print(f"Last modified: {job_mtime}")

        # Analyze job output
        try:
            with open(job_file, "r") as f:
                content = f.read()

            # Look for GPU information
            gpu_info = extract_gpu_info(content)
            if gpu_info:
                print("GPU Information:")
                for info in gpu_info:
                    print(f"  {info}")

            # Look for timing information
            timing_info = extract_timing_info(content)
            if timing_info:
                print("Timing Information:")
                for info in timing_info:
                    print(f"  {info}")

        except Exception as e:
            print(f"Error reading job file: {e}")

        # Analyze error file
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_content = f.read()

                if error_content.strip():
                    # Look for common GPU/performance related errors
                    gpu_errors = extract_gpu_errors(error_content)
                    if gpu_errors:
                        print("GPU/Performance Issues:")
                        for error in gpu_errors:
                            print(f"  {error}")

                    # Show file size as indicator of error volume
                    error_size = error_file.stat().st_size
                    print(f"Error file size: {error_size} bytes")

            except Exception as e:
                print(f"Error reading error file: {e}")


def extract_gpu_info(content):
    """Extract GPU-related information from log content."""
    gpu_info = []

    # Look for nvidia-smi output
    if "NVIDIA-SMI" in content:
        gpu_info.append("nvidia-smi output found")

    # Look for GPU model information
    gpu_model_match = re.search(r"GPU Name: (.+)", content)
    if gpu_model_match:
        gpu_info.append(f"GPU Model: {gpu_model_match.group(1)}")

    # Look for memory information
    memory_match = re.search(r"GPU Memory: (.+)", content)
    if memory_match:
        gpu_info.append(f"GPU Memory: {memory_match.group(1)}")

    return gpu_info


def extract_timing_info(content):
    """Extract timing information from log content."""
    timing_info = []

    # Look for start/end times
    start_match = re.search(r"Job started at: (.+)", content)
    if start_match:
        timing_info.append(f"Started: {start_match.group(1)}")

    # Look for training completion
    if "Training finished" in content:
        timing_info.append("Training completed successfully")

    return timing_info


def extract_gpu_errors(content):
    """Extract GPU-related errors from error log content."""
    gpu_errors = []

    # Common GPU/CUDA errors
    error_patterns = [
        r"CUDA out of memory",
        r"RuntimeError.*GPU",
        r"NVIDIA.*error",
        r"cuDNN.*error",
        r"GPU.*allocation.*failed",
    ]

    lines = content.split("\n")
    for line in lines:
        for pattern in error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                gpu_errors.append(line.strip())
                break

    return gpu_errors[:5]  # Limit to first 5 errors


if __name__ == "__main__":
    analyze_job_logs()
