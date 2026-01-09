#!/bin/bash
#
# EXAMPLE SCRIPT FOR TESTING BASE FRAMEWORK
# ==========================================
# This is a minimal example for testing that the TTM framework works.
# For production training, use scripts/training/slurm/single_gpu.sh
#
#SBATCH --job-name=test_ttm_framework
#SBATCH --output=logs/example_ttm_%j.out
#SBATCH --error=logs/example_ttm_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
##SBATCH --mail-user=cjrisi@uwaterloo.ca
##SBATCH --mail-type=ALL

echo "========================================="
echo "TTM Framework Test (Example Script)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Load any necessary modules (adjust as needed for your cluster)
# module load cuda/11.8
# module load python/3.10

# Debug: Show SLURM environment
echo "Debug - SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Debug - PWD: $PWD"

# Determine project root
# Use SLURM_SUBMIT_DIR if available (when submitted via sbatch)
# Otherwise fall back to detecting from script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    echo "✓ Using SLURM submit directory: $PROJECT_ROOT"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    echo "✓ Detected project root from script: $PROJECT_ROOT"
fi

# Navigate to project root
echo "Changing to project root..."
cd "${PROJECT_ROOT}" || { echo "❌ Failed to cd to $PROJECT_ROOT"; exit 1; }
echo "Current directory: $(pwd)"

# Activate virtual environment (check common locations)
echo "Activating virtual environment..."
if [ -f ".noctprob-venv/bin/activate" ]; then
    source .noctprob-venv/bin/activate
    echo "✓ Activated .noctprob-venv"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Activated venv"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated .venv"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Using existing virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  WARNING: No virtual environment found, using system Python"
fi

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Run the single GPU example
echo "Running single GPU TTM training example..."
python scripts/examples/example_single_gpu_ttm.py

# Capture exit code
exit_code=$?

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
