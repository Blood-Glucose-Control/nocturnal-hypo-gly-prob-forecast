#!/bin/bash
#SBATCH --job-name=timesfm_ft
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Configuration (override with --export)
: ${DATASET:="brown_2019"}
: ${EPOCHS:="10"}
: ${BATCH_SIZE:="32"}
: ${LEARNING_RATE:="1e-4"}
: ${CONTEXT_LENGTH:="512"}
: ${FORECAST_LENGTH:="72"}

echo "========================================="
echo "TimesFM Finetuning"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Started: $(date)"
echo "========================================="

cd $SLURM_SUBMIT_DIR

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".noctprob-venv/bin/activate" ]; then
    source .noctprob-venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Create logs directory
mkdir -p logs

# Run finetuning
python scripts/examples/example_finetune.py \
    --model timesfm \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --context-length "$CONTEXT_LENGTH" \
    --forecast-length "$FORECAST_LENGTH"

echo "========================================="
echo "Completed: $(date)"
echo "========================================="
