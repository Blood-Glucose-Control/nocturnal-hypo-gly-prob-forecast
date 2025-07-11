#!/bin/bash
#SBATCH --job-name=train_dl_model
#SBATCH --output=logs/%j_train_dl.out
#SBATCH --error=logs/%j_train_dl.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
#SBATCH --account=hi_group
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPU assigned: $(nvidia-smi -L 2>/dev/null || echo 'No GPU info available')"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Load any required modules (uncomment as needed)
# module load cuda/11.7
# module load python/3.9
# module load anaconda3

# Activate virtual environment
if [ -d ".noctprob-venv" ]; then
    echo "Activating virtual environment..."
    source .noctprob-venv/bin/activate
else
    echo "Virtual environment not found. Please ensure .noctprob-venv exists."
    exit 1
fi

# Verify Python and GPU availability
echo "Python version: $(python --version)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"

# Set default parameters (can be overridden by command line arguments)
CONFIG_FILE="${CONFIG_FILE:-scripts/training/configs/dl_models.yaml}"
PATIENT_IDS="${PATIENT_IDS:-}"  # Empty means all patients

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls scripts/training/configs/*.yaml 2>/dev/null || echo "No config files found"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
if [ -n "$PATIENT_IDS" ]; then
    echo "Training on specific patients: $PATIENT_IDS"
else
    echo "Training on all available patients"
fi

# Run the training script (will be train_dl.py when that's implemented)
echo "Starting deep learning model training..."
if [ -n "$PATIENT_IDS" ]; then
    python scripts/training/train_dl.py \
        --config "$CONFIG_FILE" \
        --patients $PATIENT_IDS
else
    python scripts/training/train_dl.py \
        --config "$CONFIG_FILE"
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $?"
    exit 1
fi

# Print completion info
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"

# Show GPU memory usage after training
nvidia-smi 2>/dev/null || echo "No GPU monitoring available"

# Show disk usage of output directory
if [ -d "models" ]; then
    echo "Models directory size:"
    du -sh models/
fi

exit 0
