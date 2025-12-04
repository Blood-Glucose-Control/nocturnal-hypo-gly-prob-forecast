# Setup Guide for Running Benchmark on WatGPU

## Prerequisites
- Access to WATGPU cluster (UWaterloo CS department)

## Installation Guide

### 1. Connect to WATGPU Server
```bash
ssh <your_username>@watgpu.cs.uwaterloo.ca
```
> **Note:** Replace `<your_username>` with your UWaterloo username.

### 2. Clone the Repository
```bash
# Navigate to home directory (REQUIRED - must be in HOME directory)
cd ~

# Clone the repository
git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast.git
```

### 3. Set Up Python Environment
```bash
# Enter project directory
cd nocturnal-hypo-gly-prob-forecast

# Create virtual environment with Python 3.11
python3.11 -m venv .noctprob-venv

# Activate the virtual environment
source .noctprob-venv/bin/activate
```
> **Note:** The server comes with a base conda environment that includes Python 3.11.

### 4. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install the project package in development mode
pip install -e .
```

---

## Job Submission Guidelines

### ⚠️ IMPORTANT: Server Usage Policy

    - **NEVER RUN SCRIPTS DIRECTLY ON THE WATGPU LOGIN SERVER**
    - The login server is for job submission only
    - All script execution must use `sbatch`
    - Reference: [How to submit a job](https://watgpu.cs.uwaterloo.ca/slurm.html)

### Project Structure
All scripts are located at `~/nocturnal-hypo-gly-prob-forecast/scripts/watgpu_slurm/`:

Key files:proper resource allocation

### Job Submission Process

#### 1. Configure `job.sh`

**Resource and YAML Configuration:**
```bash
declare -A job_specs=(
    ["0_naive_05min.yaml"]="1 4 02:00:00"
    ["0_naive_15min.yaml"]="1 3 02:00:00"
)
```
> Format: `[yaml_file]="cores memory(GB) time(HH:MM:SS)"`
> Note: Queue time limit is 7 days maximum

**Email Notification:**
```bash
email="your.email@example.com"
```

**Run Description:**
```bash
description="This run evaluates the impact of removing exogenous variables (IOB and COB)
to determine if there is any performance degradation compared to baseline."
```
> Add a clear explanation of:
>
>   - The purpose of this run
>   - Why you're running this experiment
>   - Key changes from previous runs

#### 2. Submit the Job
```bash
cd ~/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/
bash batch.sh
```
You'll receive a job ID after submission (e.g., `Submitted batch job 12345`)

### Results Location

**Log Files:**

    - Located in `scripts/watgpu/`
    - `JOB<jobid>.out`: Standard output
    - `JOB<jobid>.err`: Error messages

**Results Directory:**
Check `results/processed/` for a timestamped folder containing:

    - Configuration details
    - Performance metrics from different scorers
    - Folder name includes run timestamp

---

## SLURM Reference Guide

### Resource Monitoring

**CPU Status:**
```bash
sinfo -o "%C"
```
Output shows: CPUS(A/I/O/T)

    - A: Allocated (in use)
    - I: Idle (available)
    - O: Other (down/maintenance)
    - T: Total CPUs

**GPU Status:**
```bash
sinfo -o "%n %G"
```
Shows available GPUs per node

**Memory Status:**
```bash
sinfo -o "%n %m"
```
Shows memory (MB) per node

### Job Management

**View Your Jobs:**
```bash
# Basic job status
squeue -u $USER

# Detailed job information
squeue -o "%.18i %.9P %.15j %.8u %.2t %.10M %.6D %C %.6m" | grep $USER
```
> Shows: JobID, Partition, JobName, User, State, Time, Nodes, CPUs, Memory

**Control Jobs:**
```bash
# Cancel a specific job
scancel <jobid>

# Cancel all your jobs
scancel -u $USER
```
