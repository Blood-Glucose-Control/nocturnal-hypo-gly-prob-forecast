# Setup Guide for Running a Benchmark on WatGPU

## Prerequisites
- Access to WATGPU cluster (UWaterloo CS department)

## Step-by-Step Installation

### 1. Connect to WATGPU Server
```bash
ssh <your_username>@watgpu.cs.uwaterloo.ca
```
> Note: Replace `your_username` with your UWaterloo username.

### 2. Clone the Repository
Navigate to your home directory and clone the project
It has to be HOME directory for now!!
```bash
# Navigate to home directory
cd ~

# Clone the repository
git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast.git
```

### 3. Set Up Python Environment
Navigate to the project directory and create a virtual environment:
```bash
# Enter project directory
cd nocturnal-hypo-gly-prob-forecast

# Create virtual environment with Python 3.11
python3.11 -m venv .noctprob-venv

# Activate the virtual environment
source .noctprob-venv/bin/activate
```
> Note: When you first ssh to the server, there is a base conda env and it came with python 11

### 4. Install Dependencies
With the virtual environment activated, install the required packages:
```bash
# Install required packages
pip install -r requirements.txt

# Install the project package in development mode
pip install -e .
```

## How to Submit a Job

### ⚠️ Critical Server Usage Policy
- **DO NOT RUN ANY SCRIPT ON THE WATGPU LOGIN SERVER DIRECTLY**
- The login server is purely for job submission (the server you SSH into)
- All script execution must be done through job submission using `sbatch`
- Source: [How to submit a job](https://watgpu.cs.uwaterloo.ca/slurm.html)

### Project Structure
All of our scripts are located at `~/nocturnal-hypo-gly-prob-forecast/scripts/watgpu`:

```bash
cd ~/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/
```

Key files:
1. `job.sh`: Configure resources for a "model run" (a single YAML configuration)
2. `batch.sh`: Submit multiple model runs in a single job. Each yaml files will be submitted with the resource as one job.
3. `run_model.py`: Entry point for the benchmark (configuration settings)

### Run Checklist

1. **Configure resources in `job.sh`**
   - Review and adjust resource settings:
     - `--time`: Maximum runtime (format: HH:MM:SS)
     - `--mem`: Memory allocation (e.g., 16G)
     - `--cpus-per-task`: Number of CPU cores
   - Update your email notification settings:
     ```bash
     #SBATCH --mail-user=<your_email>@uwaterloo.ca
     ```

2. **Configure YAML files in `batch.sh`**
   - Add all YAML configuration files you want to run
   - All YAML files should be located in `src/runing/configs/`
   - Each yaml uses the resource you specified in `job.sh` so if it is 5 cores and 3 files, then you would have 15 cores acquired. So.. Be cautious!
   - Example:
     ```bash
     yaml_files=(
        "0_naive_05min.yaml",
        "0_naive_15min.yaml",
        "2_arch_05min.yaml"
     )
     ```

3. **Document your run**
   - Add a clear description in `batch.sh` that includes:
     - The main purpose of the run
     - Why you're running this experiment
     - Key changes from previous runs
   - Example:
     ```bash
     # Run description:
      msg="This run evaluates the impact of removing exogenous variables (IOB and COB)
      to determine if there is any performance degradation compared to baseline."
     ```

4. **Submit the job**
   - Navigate to the scripts directory:
     ```bash
     cd ~/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/
     ```
   - Submit the job using:
     ```bash
     bash batch.sh
     ```
   - Note the job ID that appears after submission (e.g., `Submitted batch job 12345`)

### Results
You will receive email notifications about job status (start, end, or failure) if configured correctly.

1. **Log Files**: In `scripts/watgpu/`, you'll find two files with the job ID:
   - `JOB<jobid>.out`: Standard output from the run
   - `JOB<jobid>.err`: Error messages (if any)

2. **Results Directory**: Check `results/processed/` for a timestamped folder containing:
   - Configuration details
   - Performance metrics from different scorers
   - The folder name includes the timestamp of when the job was run

### SLURM Quick Reference Guide

Source: [https://watgpu.cs.uwaterloo.ca/slurm.html#slurm](https://watgpu.cs.uwaterloo.ca/slurm.html#slurm)

## Basic Resource Monitoring

### CPU Status
```bash
sinfo -o "%C"
```
Output shows: CPUS(A/I/O/T)
- A: Allocated (in use)
- I: Idle (available)
- O: Other (down/maintenance)
- T: Total CPUs

### GPU Status
```bash
sinfo -o "%n %G"
```
Shows available GPUs per node

### Memory Status
```bash
sinfo -o "%n %m"
```
Shows memory (MB) per node

## Job Management

### View Jobs
```bash
# Check your jobs
squeue -u $USER

# Check all jobs with details
squeue -o "%.18i %.9P %.15j %.8u %.2t %.10M %.6D %C %.6m" | grep $USER
# Shows: JobID, Partition, JobName, User, State, Time, Nodes, CPUs, Mem
```

### Control Jobs
```bash
# Cancel a specific job
scancel <jobid>

# Cancel all your jobs
scancel -u $USER
```
