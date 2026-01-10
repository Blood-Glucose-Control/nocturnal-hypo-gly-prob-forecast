#!/bin/bash
#
# SBATCH SCRIPT FOR FINETUNING TOTO
# ==========================================
# Optimized for single-GPU iterative testing.
#
#SBATCH --job-name=finetune_toto_diabetes
#SBATCH --output=logs/toto_ft_%j.out
#SBATCH --error=logs/toto_ft_%j.err
#SBATCH --time=02:00:00            # Increased slightly for Toto's larger params
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB                 # Toto's 151M params + 4096 context need more RAM
#SBATCH --gres=gpu:1               # Still using 1 GPU for iteration
#SBATCH --partition=HI

echo "========================================="
echo "Toto Finetuning Job Started"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# 1. Project Root Detection
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_ROOT}" || exit 1

# 2. Virtual Environment Activation
source .noctprob-venv/bin/activate || source venv/bin/activate || source .venv/bin/activate

# 3. GPU Diagnostics
# Toto benefits from Flash Attention; checking if your GPU supports it (Ampere+)
nvidia-smi

# 4. Run Toto Finetuning
# Note: Toto uses "MaskedTimeseries" objects and needs specific context lengths.
echo "Starting Toto finetuning pipeline..."
python scripts/examples/example_single_gpu_toto.py

# Capture exit code
exit_code=$?

echo "========================================="
echo "Job completed at: $(date) with code: $exit_code"
exit $exit_code
