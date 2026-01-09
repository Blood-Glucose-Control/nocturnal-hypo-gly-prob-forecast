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

# Activate your virtual environment
echo "Activating virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${NOCTURNAL_VENV:=${PROJECT_ROOT}/.noctprob-venv/bin/activate}"
# shellcheck disable=SC1090
source "${NOCTURNAL_VENV}"

# Navigate to project root
cd "${PROJECT_ROOT}"

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
