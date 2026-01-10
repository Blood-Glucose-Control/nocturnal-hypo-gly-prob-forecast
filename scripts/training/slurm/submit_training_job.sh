#!/bin/bash

# Enhanced SLURM job submission script for RL Insulin Pump training
# Creates a self-contained run directory with all outputs organized

# Default values
TIME="01:00:00"
MEM="8G"
CPUS=4
GRES="gpu:1"
PARTITION="HI"
ENV="env"
CONFIG=""
RUN_NAME=""

RUN_PREDICT=false
EMAIL=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        '--config') CONFIG="$2"; shift 2;;
        '--run-name') RUN_NAME="$2"; shift 2;;
        '--time') TIME="$2"; shift 2;;
        '--mem') MEM="$2"; shift 2;;
        '--cpus') CPUS="$2"; shift 2;;
        '--gres') GRES="$2"; shift 2;;
        '--partition') PARTITION="$2"; shift 2;;
        '--env') ENV="$2"; shift 2;;
        '--email') EMAIL="$2"; shift 2;;
        '--predict') RUN_PREDICT=true; shift;;
        '--help')
            echo "Usage: $0 --config path/to/config.yaml [options]"
            echo "Options:"
            echo "  --config       Path to YAML configuration file (required)"
            echo "  --run-name     Name for this training run (auto-generated if not provided)"
            echo "  --time         SLURM time limit (default: 01:00:00)"
            echo "  --mem          Memory limit (default: 8G)"
            echo "  --cpus         Number of CPUs (default: 4)"
            echo "  --gres         GPU resources (default: gpu:1)"
            echo "  --partition    SLURM partition (default: HI)"
            echo "  --env          Virtual environment path (default: env)"
            echo "  --email        Email for notifications (optional)"
            echo "  --predict      Run prediction after training (optional)"
            echo "  --help         Show this help message"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Validate required arguments
# if [[ -z "$CONFIG" ]]; then
#     echo "Error: --config is required"
#     echo "Use --help for usage information"
#     exit 1
# fi

# if [[ ! -f "$CONFIG" ]]; then
#     echo "Error: Configuration file '$CONFIG' does not exist"
#     exit 1
# fi

# Generate run name if not provided
if [[ -z "$RUN_NAME" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    RUN_NAME="${CONFIG_NAME}_${TIMESTAMP}"
fi

# Create run directory structure
BASE_DIR="/u201/y329xie/nocturnal-hypo-gly-prob-forecast"
RUN_DIR="$BASE_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"/{logs,checkpoints,tensorboard,results,configs,slurm}
mkdir -p "$RUN_DIR"/results/{train,predict,plots}
echo "Creating training run: $RUN_NAME"
echo "Run directory: $RUN_DIR"

# Copy configuration to run directory
# cp "$CONFIG" "$RUN_DIR/configs/"
# CONFIG_COPY="$RUN_DIR/configs/$(basename "$CONFIG")"

# Create modified config that points to run-specific directories
# python3 << EOF
# import yaml
# import sys

# # Load original config
# with open('$CONFIG', 'r') as f:
#     config = yaml.safe_load(f)

# # Update paths to point to run directory
# config['run_directory'] = '$RUN_DIR'

# # Save modified config
# with open('$CONFIG_COPY', 'w') as f:
#     yaml.dump(config, f, default_flow_style=False)

# print("Updated configuration saved to $CONFIG_COPY")
# EOF

# Create SLURM job script
JOB_SCRIPT="$RUN_DIR/slurm/job_script.sh"
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=rl-insulin-$RUN_NAME
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=$GRES
#SBATCH --partition=$PARTITION
#SBATCH --output=$RUN_DIR/slurm/slurm-%j.out
#SBATCH --error=$RUN_DIR/slurm/slurm-%j.err
EOF

# Add email notifications if provided
if [[ -n "$EMAIL" ]]; then
    cat >> "$JOB_SCRIPT" << EOF
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=ALL
EOF
fi


# Add job execution commands
cat >> "$JOB_SCRIPT" << EOF

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Run directory: $RUN_DIR"
echo "Configuration: $CONFIG_COPY"

# Change to project directory
cd $BASE_DIR

# Activate virtual environment
echo "Activating virtual environment: $ENV"
source $ENV/bin/activate

# Save system information
echo "Python version: \$(python --version)" > $RUN_DIR/logs/system_info.txt
echo "GPU information:" >> $RUN_DIR/logs/system_info.txt
nvidia-smi >> $RUN_DIR/logs/system_info.txt 2>/dev/null || echo "No GPU available" >> $RUN_DIR/logs/system_info.txt
echo "Environment packages:" >> $RUN_DIR/logs/system_info.txt
pip list >> $RUN_DIR/logs/system_info.txt

# Save git information if available
if [[ -d .git ]]; then
    echo "Git commit: \$(git rev-parse HEAD)" >> $RUN_DIR/logs/system_info.txt
    echo "Git branch: \$(git branch --show-current)" >> $RUN_DIR/logs/system_info.txt
    echo "Git status:" >> $RUN_DIR/logs/system_info.txt
    git status --porcelain >> $RUN_DIR/logs/system_info.txt
fi


# Run the training
echo "Starting training..."
python /u201/y329xie/AdaPTS/scripts/run.py --forecast_horizon 24 --model_name "Autogluon/chronos-t5-small" --context_length 512 --seed $RANDOM --device "cuda:0" --dataset_name "Illness" --data-path $RUN_DIR/results/latest.csv --adapter "linearlVAE" --use_revin --supervised "True"

# # Optionally run prediction after training
# if [[ "$RUN_PREDICT" == true ]]; then
#     echo "Running prediction after training..."
#     python src/main.py --cfg "$CONFIG_COPY" --mode predict
# fi

# # Save final model with run-specific name
# if [[ -f "ddpg_simglucose.zip" ]]; then
#     mv ddpg_simglucose.zip "$RUN_DIR/checkpoints/final_model.zip"
# fi

# Move any other generated files to run directory
for file in *.pth *.txt _stable_baselines3_version; do
    if [[ -f "$file" ]]; then
        mv "$file" "$RUN_DIR/checkpoints/"
    fi
done

echo "Job completed at: \$(date)"
echo "All outputs saved to: $RUN_DIR"
EOF

# Make job script executable
chmod +x "$JOB_SCRIPT"

# Create run summary
cat > "$RUN_DIR/run_summary.txt" << EOF
Training Run: $RUN_NAME
Created: $(date)
Configuration: $(basename "$CONFIG")
Original config: $CONFIG

SLURM Settings:
- Time limit: $TIME
- Memory: $MEM
- CPUs: $CPUS
- GPU: $GRES
- Partition: $PARTITION

Directory Structure:
- Run directory: $RUN_DIR
- Logs: $RUN_DIR/logs/
- Checkpoints: $RUN_DIR/checkpoints/
- TensorBoard: $RUN_DIR/tensorboard/
- Results: $RUN_DIR/results/
- SLURM logs: $RUN_DIR/slurm/

To monitor the job:
- squeue -u $(whoami)
- tail -f $RUN_DIR/slurm/slurm-*.out

To view TensorBoard:
- tensorboard --logdir $RUN_DIR/tensorboard/
EOF

echo ""
echo "Run summary:"
cat "$RUN_DIR/run_summary.txt"

# Submit the job
echo ""
echo "Submitting job..."
sbatch "$JOB_SCRIPT"

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f $RUN_DIR/slurm/slurm-*.out"
