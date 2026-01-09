#!/bin/bash
#
# PRODUCTION SINGLE GPU TRAINING
# ================================
# Use this for production model training on a single GPU
#
# Quick Start:
#   sbatch scripts/training/slurm/single_gpu.sh
#
# With custom config:
#   sbatch --export=CONFIG_PATH=configs/models/ttm/custom.yaml scripts/training/slurm/single_gpu.sh
#
#SBATCH --job-name=ttm_train_1gpu
#SBATCH --output=logs/train_1gpu_%j.out
#SBATCH --error=logs/train_1gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --mail-user=your.email@example.com
##SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration (can be overridden with --export)
: ${CONFIG_PATH:="configs/models/ttm/fine_tune.yaml"}
: ${DATA_CONFIG:="configs/data/kaggle_bris_t1d.yaml"}
: ${OUTPUT_DIR:="trained_models/artifacts/ttm"}
: ${EXPERIMENT_NAME:="single_gpu_training"}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "========================================="
echo "TTM Single GPU Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  Model config: $CONFIG_PATH"
echo "  Data config: $DATA_CONFIG"
echo "  Output dir: $OUTPUT_DIR"
echo "  Experiment: $EXPERIMENT_NAME"
echo "========================================="
echo ""

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Determine project root (script is in scripts/training/slurm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Project root: $PROJECT_ROOT"

# Navigate to project root
cd "$PROJECT_ROOT"

# Activate virtual environment (check common locations)
echo "Activating environment..."
if [ -f ".noctprob-venv/bin/activate" ]; then
    source .noctprob-venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using existing virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  WARNING: No virtual environment found, using system Python"
fi

# =============================================================================
# HARDWARE VERIFICATION
# =============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo ""

# =============================================================================
# TRAINING
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Set environment variables for optimal single GPU performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting training..."
echo "Command: python -m src.training.train_model --config $CONFIG_PATH"
echo ""

# Run training
# TODO: Replace with actual unified training script when implemented
# For now, use the TTM example
python scripts/examples/example_single_gpu_ttm.py \
    --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
    2>&1 | tee "logs/train_${SLURM_JOB_ID}.log"

# Capture exit code
exit_code=$?

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "========================================="
echo "Training completed: $(date)"
echo "Exit code: $exit_code"
echo "Duration: $SECONDS seconds"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ SUCCESS: Model saved to $OUTPUT_DIR/$EXPERIMENT_NAME"
else
    echo "❌ FAILED: Check logs/train_${SLURM_JOB_ID}.log for details"
fi

exit $exit_code
