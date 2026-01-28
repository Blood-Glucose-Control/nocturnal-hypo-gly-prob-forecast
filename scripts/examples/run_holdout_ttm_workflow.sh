#!/bin/bash
#
# HOLDOUT SYSTEM WORKFLOW WITH TTM TRAINING (COMBINED DATASETS)
# ==============================================================
# Complete end-to-end workflow: combines multiple datasets, trains on mixed data
#
# Quick Start:
#   sbatch scripts/examples/run_holdout_ttm_workflow.sh
#
# With specific datasets:
#   sbatch --export=DATASETS="lynch_2022 brown_2019" scripts/examples/run_holdout_ttm_workflow.sh
#
# All datasets combined:
#   sbatch --export=DATASETS="lynch_2022 aleppo brown_2019 tamborlane_2008" scripts/examples/run_holdout_ttm_workflow.sh
#
#SBATCH --job-name=holdout_ttm
#SBATCH --output=trained_models/logs/holdout_ttm_%j.out
#SBATCH --error=trained_models/logs/holdout_ttm_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
#SBATCH --mail-user=cjrisi@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration (can be overridden with --export)
# DATASETS can be space-separated list: "lynch_2022 aleppo brown_2019"
: ${DATASETS:="tamborlane_2008 brown_2019"}
: ${CONFIG_DIR:="configs/data/holdout_5pct"}
: ${OUTPUT_BASE_DIR:="trained_models/artifacts/_tsfm_testing/$(date +%Y-%m-%d_%H:%M)_JID${SLURM_JOB_ID}_holdout_workflow"}
: ${SKIP_TRAINING:="true"}
: ${EPOCHS:="1"}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "========================================="
echo "Holdout System Workflow with TTM"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  Datasets: $DATASETS"
echo "  Config dir: $CONFIG_DIR"
echo "  Output base dir: $OUTPUT_BASE_DIR"
echo "  Skip training: $SKIP_TRAINING"
echo "  Epochs: $EPOCHS"
echo "========================================="
echo ""

# Determine project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    echo "✓ Using SLURM submit directory: $PROJECT_ROOT"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "✓ Detected project root from script: $PROJECT_ROOT"
fi

# Navigate to project root
echo "Changing to project root..."
cd "$PROJECT_ROOT" || { echo "❌ Failed to cd to $PROJECT_ROOT"; exit 1; }
echo "Current directory: $(pwd)"

# Activate virtual environment
echo "Activating environment..."
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

# Verify Python
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# =============================================================================
# HARDWARE VERIFICATION
# =============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo ""

# =============================================================================
# RUN WORKFLOW
# =============================================================================

# Create output and log directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "trained_models/logs"

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Track overall status
overall_exit_code=0

echo ""
echo "==================================================================="
echo "# Training on combined datasets: $DATASETS"
echo "==================================================================="
echo ""

# Build command - pass all datasets together
CMD="python scripts/examples/example_holdout_ttm_workflow.py"
CMD="$CMD --datasets $DATASETS"
CMD="$CMD --config-dir $CONFIG_DIR"
CMD="$CMD --output-dir $OUTPUT_BASE_DIR"
CMD="$CMD --epochs $EPOCHS"
if [ "$SKIP_TRAINING" = "true" ]; then
    CMD="$CMD --skip-training"
fi

echo "Running combined workflow..."
# Pretty print for logs (display only)
echo "Command:"
echo "  python scripts/examples/example_holdout_ttm_workflow.py \\"
echo "    --datasets $DATASETS \\"
echo "    --config-dir $CONFIG_DIR \\"
echo "    --output-dir $OUTPUT_BASE_DIR \\"
echo "    --epochs $EPOCHS \\"
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "    --skip-training"
else
    echo ""
fi
echo ""

# Run the workflow script with all datasets
$CMD 2>&1 | tee "trained_models/logs/holdout_workflow_combined_${SLURM_JOB_ID}.log"

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: Combined training completed"
else
    echo ""
    echo "❌ FAILED: Combined training failed with exit code $exit_code"
    overall_exit_code=1
fi

echo ""
echo "-------------------------------------------------------------------"

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "========================================="
echo "All Workflows Completed: $(date)"
echo "Total duration: $SECONDS seconds"
echo "========================================="
echo ""

# Move logs to output directory for easy access
echo "Moving logs to output directory..."
mv "trained_models/logs/holdout_ttm_${SLURM_JOB_ID}.out" "${OUTPUT_BASE_DIR}/" 2>/dev/null || echo "  (stdout log not found yet)"
mv "trained_models/logs/holdout_ttm_${SLURM_JOB_ID}.err" "${OUTPUT_BASE_DIR}/" 2>/dev/null || echo "  (stderr log not found yet)"
mv "trained_models/logs/holdout_workflow_combined_${SLURM_JOB_ID}.log" "${OUTPUT_BASE_DIR}/" 2>/dev/null

# Print summary
if [ $overall_exit_code -eq 0 ]; then
    echo "✅ Combined training successful"
    echo "   Model: ${OUTPUT_BASE_DIR}/model.pt"
    echo "   Logs: ${OUTPUT_BASE_DIR}/"
    echo "     - holdout_ttm_${SLURM_JOB_ID}.out (SLURM stdout)"
    echo "     - holdout_ttm_${SLURM_JOB_ID}.err (SLURM stderr)"
    echo "     - holdout_workflow_combined_${SLURM_JOB_ID}.log (workflow log)"
else
    echo "❌ Combined training failed"
    echo "   Logs: ${OUTPUT_BASE_DIR}/"
fi

echo ""
echo "All outputs saved to: $OUTPUT_BASE_DIR/"
echo "========================================="

exit $overall_exit_code
