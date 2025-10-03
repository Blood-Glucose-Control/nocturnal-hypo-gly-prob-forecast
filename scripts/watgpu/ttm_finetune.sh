#!/bin/bash

#SBATCH --job-name="ttm_finetune"
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH -o /u6/cjrisi/nocturnal/results/runs/ttm_finetune/run_%j_%x.out
#SBATCH -e /u6/cjrisi/nocturnal/results/runs/ttm_finetune/run_%j_%x.err
#SBATCH --mail-user=cjrisi@uwaterloo.ca,t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# TTM Fine-tuning SLURM Script with YAML Configuration Support
# Usage: sbatch ttm_finetune.sh [path/to/config.yaml]
#
# Example: sbatch ttm_finetune.sh models/configs/ttm_baseline_config.yaml

# Parse command line arguments
CONFIG_FILE="${1:-}"

# Validate configuration file if provided
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Configuration file '$CONFIG_FILE' not found!"
        exit 1
    fi
    echo "Using configuration file: $CONFIG_FILE"
    CONFIG_FILE=$(realpath "$CONFIG_FILE")  # Convert to absolute path
else
    echo "No configuration file provided, using default parameters"
fi

# Create timestamped run directory
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/nocturnal/results/runs/ttm_finetune/run_${RUN_TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate

# Record precise start time
RUN_START_TIME=$(date '+%s')
RUN_START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

# Create run info file with SLURM resource settings
echo "=== Run Information ===" > "$RUN_DIR/run_info.txt"
echo "Job ID: $SLURM_JOB_ID" >> "$RUN_DIR/run_info.txt"
echo "Job Name: $SLURM_JOB_NAME" >> "$RUN_DIR/run_info.txt"
echo "Node: $SLURMD_NODENAME" >> "$RUN_DIR/run_info.txt"
echo "Start Time: $RUN_START_TIMESTAMP" >> "$RUN_DIR/run_info.txt"
echo "Start Time (Unix): $RUN_START_TIME" >> "$RUN_DIR/run_info.txt"
echo "Run Directory: $RUN_DIR" >> "$RUN_DIR/run_info.txt"
echo "" >> "$RUN_DIR/run_info.txt"

# SLURM Resource Configuration
echo "=== SLURM Resource Settings ===" >> "$RUN_DIR/run_info.txt"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK" >> "$RUN_DIR/run_info.txt"
echo "Memory per Node: $SLURM_MEM_PER_NODE MB" >> "$RUN_DIR/run_info.txt"
echo "Memory per CPU: $SLURM_MEM_PER_CPU MB" >> "$RUN_DIR/run_info.txt"
echo "Partition: $SLURM_JOB_PARTITION" >> "$RUN_DIR/run_info.txt"
echo "Time Limit: $SLURM_TIMELIMIT" >> "$RUN_DIR/run_info.txt"
echo "QOS: $SLURM_JOB_QOS" >> "$RUN_DIR/run_info.txt"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES" >> "$RUN_DIR/run_info.txt"
echo "Number of Tasks: $SLURM_NTASKS" >> "$RUN_DIR/run_info.txt"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE" >> "$RUN_DIR/run_info.txt"
echo "GPU Type: $SLURM_JOB_GPUS" >> "$RUN_DIR/run_info.txt"
echo "Account: $SLURM_JOB_ACCOUNT" >> "$RUN_DIR/run_info.txt"
echo "Working Directory: $SLURM_SUBMIT_DIR" >> "$RUN_DIR/run_info.txt"
echo "Config File: ${CONFIG_FILE:-none}" >> "$RUN_DIR/run_info.txt"
echo "========================" >> "$RUN_DIR/run_info.txt"
echo "" >> "$RUN_DIR/run_info.txt"

# Capture actual node resources
echo "=== Node Hardware Information ===" >> "$RUN_DIR/run_info.txt"
echo "Total CPUs on Node: $(nproc --all)" >> "$RUN_DIR/run_info.txt"
echo "Total Memory on Node: $(free -h | grep Mem: | awk '{print $2}')" >> "$RUN_DIR/run_info.txt"
echo "Available Memory on Node: $(free -h | grep Mem: | awk '{print $7}')" >> "$RUN_DIR/run_info.txt"
echo "CPU Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)" >> "$RUN_DIR/run_info.txt"
echo "CPU Architecture: $(lscpu | grep 'Architecture' | cut -d: -f2 | xargs)" >> "$RUN_DIR/run_info.txt"
echo "CPU Sockets: $(lscpu | grep 'Socket(s)' | cut -d: -f2 | xargs)" >> "$RUN_DIR/run_info.txt"
echo "Cores per Socket: $(lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs)" >> "$RUN_DIR/run_info.txt"
echo "Threads per Core: $(lscpu | grep 'Thread(s) per core' | cut -d: -f2 | xargs)" >> "$RUN_DIR/run_info.txt"
echo "=====================================" >> "$RUN_DIR/run_info.txt"
echo "" >> "$RUN_DIR/run_info.txt"

# Debug: Check environment
echo "=== Environment Check ==="
echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Run Directory: $RUN_DIR"
which python
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"

# GPU Information
echo "=== GPU Information ==="
nvidia-smi
echo "GPU Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
echo "GPU Name: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
echo "==========================="

# Register run start in model registry
echo "Registering run in model registry..."
python -c "
import sys
sys.path.append('$HOME/nocturnal/results/runs')
from model_registry import ModelRegistry
import yaml

# Load config if available
config = None
if '$CONFIG_FILE' and '$CONFIG_FILE' != '':
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)

# Collect SLURM and hardware info
slurm_info = {
    'node_name': '$SLURMD_NODENAME',
    'cpus_per_task': '$SLURM_CPUS_PER_TASK',
    'mem_per_cpu_gb': '${SLURM_MEM_PER_CPU:-}',  
    'partition': '$SLURM_JOB_PARTITION',
    'time_limit': '$SLURM_TIMELIMIT'
}

hardware_info = {
    'cpu_count': '$(nproc --all)',
    'total_memory_gb': '$(free -g | grep Mem: | awk \"{print \$2}\")',
    'gpu_type': '$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)',
    'gpu_memory_gb': '$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk \"{print \$1/1024}\")'
}

registry = ModelRegistry()
run_id = registry.register_run_start('$RUN_DIR', config, slurm_info, hardware_info)
print(f'Registered run: {run_id}')
" || echo "Warning: Could not register run start"

# Start GPU monitoring in background  
nvidia-smi dmon -s pucvmet -d 10 -o DT > "$RUN_DIR/gpu_monitoring.log" &
GPU_MONITOR_PID=$!

# Log GPU utilization periodically
(
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits)" >> "$RUN_DIR/gpu_utilization.log"
        sleep 30
    done
) &
GPU_UTIL_PID=$!

echo "Started GPU monitoring (PID: $GPU_MONITOR_PID) and utilization logging (PID: $GPU_UTIL_PID)"

# Start the training run
echo "=== Starting Training ==="
echo "Training started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

# Build python command with optional config file
PYTHON_CMD="python $HOME/nocturnal/src/train/ttm.py --run-dir \"$RUN_DIR\""
if [[ -n "$CONFIG_FILE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --config \"$CONFIG_FILE\""
fi

echo "Running command: $PYTHON_CMD"
eval $PYTHON_CMD 2>&1 | tee "$RUN_DIR/training.log"

TRAIN_EXIT_CODE=$?

# Record precise end time and calculate elapsed time
RUN_END_TIME=$(date '+%s')
RUN_END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_TIME=$((RUN_END_TIME - RUN_START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINUTES=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

echo "Training finished at: $RUN_END_TIMESTAMP"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "Total elapsed time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s ($ELAPSED_TIME seconds)"

# Stop GPU monitoring
kill $GPU_MONITOR_PID $GPU_UTIL_PID 2>/dev/null || true

# Update run info with timing results
echo "=== Timing Results ===" >> "$RUN_DIR/run_info.txt"
echo "End Time: $RUN_END_TIMESTAMP" >> "$RUN_DIR/run_info.txt"
echo "End Time (Unix): $RUN_END_TIME" >> "$RUN_DIR/run_info.txt"
echo "Total Elapsed Time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s" >> "$RUN_DIR/run_info.txt"
echo "Total Elapsed Seconds: $ELAPSED_TIME" >> "$RUN_DIR/run_info.txt"
echo "Training Exit Code: $TRAIN_EXIT_CODE" >> "$RUN_DIR/run_info.txt"
echo "" >> "$RUN_DIR/run_info.txt"

# Final GPU state
echo "=== Final GPU State ===" >> "$RUN_DIR/run_info.txt"
nvidia-smi >> "$RUN_DIR/run_info.txt"

echo "Run completed. Logs and monitoring data saved to: $RUN_DIR"

# Create run summary for registry
echo "=== Run Summary ===" >> "$RUN_DIR/run_info.txt"
echo "Exit Code: $TRAIN_EXIT_CODE" >> "$RUN_DIR/run_info.txt"
echo "Status: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo 'completed' || echo 'failed')" >> "$RUN_DIR/run_info.txt"
echo "=====================================" >> "$RUN_DIR/run_info.txt"

# Update model registry with completion info
echo "Updating model registry with completion info..."
python -c "
import sys
sys.path.append('$HOME/nocturnal/results/runs')
from model_registry import ModelRegistry

registry = ModelRegistry()
run_id = '$(basename $RUN_DIR)'
status = 'completed' if $TRAIN_EXIT_CODE == 0 else 'failed'

# Try to extract results from training log if available
results = {}
try:
    with open('$RUN_DIR/training.log', 'r') as f:
        log_content = f.read()
    # Extract final loss values if available
    # This would need custom parsing based on your log format
    pass
except:
    pass

registry.register_run_completion(run_id, results, status)
print(f'Updated registry for run: {run_id} with status: {status}')
" || echo "Warning: Could not update registry"

echo "All run data and logs saved to: $RUN_DIR"
echo "Configuration and results tracked in model registry"

# Exit with the same code as the training script
exit $TRAIN_EXIT_CODE

# Run sbatch ttm_finetune.sh to finetune the model in the terminal
