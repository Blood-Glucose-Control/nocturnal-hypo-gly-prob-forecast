#!/bin/bash
#
# HOLDOUT SYSTEM WORKFLOW - LOCAL/NON-SLURM VERSION
# ==================================================
# Complete end-to-end workflow for local development and testing.
# This is a non-SLURM version of run_holdout_ttm_workflow.sh
#
# Quick Start (uses defaults: MODEL_TYPE=ttm, CONFIG_DIR=configs/data/holdout_5pct):
#   ./scripts/examples/run_holdout_generic_workflow.sh
#
# With specific datasets:
#   DATASETS="lynch_2022 brown_2019" ./scripts/examples/run_holdout_generic_workflow.sh
#
# With specific model type:
#   MODEL_TYPE="chronos" ./scripts/examples/run_holdout_generic_workflow.sh
#   MODEL_TYPE="moment" DATASETS="lynch_2022 aleppo_2017" ./scripts/examples/run_holdout_generic_workflow.sh
#
# With specific config directory (e.g., different holdout percentage):
#   CONFIG_DIR="configs/data/holdout_10pct" ./scripts/examples/run_holdout_generic_workflow.sh
#   CONFIG_DIR="configs/data/holdout" DATASETS="brown_2019" ./scripts/examples/run_holdout_generic_workflow.sh
#
# Combining model type and config location:
#   MODEL_TYPE="ttm" CONFIG_DIR="configs/data/holdout_10pct" ./scripts/examples/run_holdout_generic_workflow.sh
#   MODEL_TYPE="chronos" CONFIG_DIR="configs/data/holdout" DATASETS="lynch_2022 brown_2019" ./scripts/examples/run_holdout_generic_workflow.sh
#
# With specific GPU (useful on multi-GPU machines):
#   CUDA_VISIBLE_DEVICES=0 ./scripts/examples/run_holdout_generic_workflow.sh
#   CUDA_VISIBLE_DEVICES=1 MODEL_TYPE="chronos" ./scripts/examples/run_holdout_generic_workflow.sh
#
# All datasets combined with specific model:
#   CUDA_VISIBLE_DEVICES=1 MODEL_TYPE="ttm" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="lynch_2022 aleppo_2017 brown_2019 tamborlane_2008" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh

# =============================================================================
# CONFIGURATION
# =============================================================================

# Generate a unique run ID (replaces SLURM_JOB_ID for local runs)
# Uses timestamp + random suffix for uniqueness
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

# Default configuration (can be overridden via environment variables)
# DATASETS can be space-separated list: "lynch_2022 aleppo brown_2019"
: ${DATASETS:="tamborlane_2008 brown_2019"}
: ${CONFIG_DIR:="configs/data/holdout_5pct"}
: ${OUTPUT_BASE_DIR:="trained_models/artifacts/_tsfm_testing/$(date +%Y-%m-%d_%H:%M)_RID${RUN_ID}_holdout_workflow"}
: ${SKIP_TRAINING:="true"}
: ${EPOCHS:="20"}
: ${BATCH_SIZE:="4096"}
: ${MODEL_TYPE:="ttm"}  # Model type: ttm, chronos, moment, etc.

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "========================================="
echo "Holdout System Workflow (Local/Non-SLURM)"
echo "========================================="
echo "Run ID: $RUN_ID"
echo "Started: $(date)"
echo "Hostname: $(hostname)"
echo ""
echo "Configuration:"
echo "  Model type: $MODEL_TYPE"
echo "  Datasets: $DATASETS"
echo "  Config dir: $CONFIG_DIR"
echo "  Output base dir: $OUTPUT_BASE_DIR"
echo "  Skip training: $SKIP_TRAINING"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "========================================="
echo ""

# Determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "✓ Detected project root from script: $PROJECT_ROOT"

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
# HARDWARE DETECTION
# =============================================================================

echo ""
echo "Hardware Detection:"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv 2>/dev/null || echo "  No NVIDIA GPU detected or driver issue"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Ensure CUDA indices match nvidia-smi ordering
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    echo "  CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "  No NVIDIA GPU detected, will use CPU"
    export CUDA_VISIBLE_DEVICES=""
fi
echo ""

# Set optimal threading based on available CPUs
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_COUNT}
echo "CPU cores available: $CPU_COUNT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# =============================================================================
# RUN WORKFLOW
# =============================================================================

# Create output and log directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "trained_models/logs"

# Track timing
START_TIME=$SECONDS

# Track overall status
overall_exit_code=0

echo ""
echo "==================================================================="
echo "# Training on combined datasets: $DATASETS"
echo "# Model type: $MODEL_TYPE"
echo "==================================================================="
echo ""

# Build command - pass all datasets together
CMD="python scripts/examples/example_holdout_generic_workflow.py"
CMD="$CMD --model-type $MODEL_TYPE"
CMD="$CMD --datasets $DATASETS"
CMD="$CMD --config-dir $CONFIG_DIR"
CMD="$CMD --output-dir $OUTPUT_BASE_DIR"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
if [ "$SKIP_TRAINING" = "true" ]; then
    CMD="$CMD --skip-training"
fi

echo "Running combined workflow..."
# Pretty print for logs (display only)
echo "Command:"
echo "  python scripts/examples/example_holdout_generic_workflow.py \\"
echo "    --model-type $MODEL_TYPE \\"
echo "    --datasets $DATASETS \\"
echo "    --config-dir $CONFIG_DIR \\"
echo "    --output-dir $OUTPUT_BASE_DIR \\"
echo "    --epochs $EPOCHS \\"
echo "    --batch-size $BATCH_SIZE \\"
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "    --skip-training"
else
    echo ""
fi
echo ""

# Create log filename with run ID (matches SLURM pattern with job ID)
LOG_FILE="trained_models/logs/holdout_workflow_${MODEL_TYPE}_${RUN_ID}.log"

# Also create stdout/stderr equivalent logs for parity with SLURM
STDOUT_LOG="trained_models/logs/holdout_${MODEL_TYPE}_${RUN_ID}.out"
STDERR_LOG="trained_models/logs/holdout_${MODEL_TYPE}_${RUN_ID}.err"

# Run the workflow script with all datasets
# Capture combined output to log file while showing to terminal
# Also save stdout and stderr separately for SLURM parity
{
    $CMD 2>&1
    echo $? > /tmp/exit_code_${RUN_ID}
} | tee "$LOG_FILE"

# Capture exit code from the python command (not tee)
exit_code=$(cat /tmp/exit_code_${RUN_ID} 2>/dev/null || echo 1)
rm -f /tmp/exit_code_${RUN_ID}

# Copy the combined log as both .out and .err for SLURM parity
# (In local runs, stdout/stderr are combined unlike SLURM which separates them)
cp "$LOG_FILE" "$STDOUT_LOG" 2>/dev/null
touch "$STDERR_LOG"  # Create empty stderr log (errors are in combined log)

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

ELAPSED=$((SECONDS - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "All Workflows Completed: $(date)"
echo "Total duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s ($ELAPSED seconds)"
echo "========================================="
echo ""

# Move logs to output directory for easy access (matches SLURM script behavior)
echo "Moving logs to output directory..."
mv "$STDOUT_LOG" "${OUTPUT_BASE_DIR}/" 2>/dev/null || echo "  (stdout log not found)"
mv "$STDERR_LOG" "${OUTPUT_BASE_DIR}/" 2>/dev/null || echo "  (stderr log not found)"
mv "$LOG_FILE" "${OUTPUT_BASE_DIR}/" 2>/dev/null || echo "  (workflow log not found)"

# Print summary
if [ $overall_exit_code -eq 0 ]; then
    echo "✅ Combined training successful"
    echo "   Model: ${OUTPUT_BASE_DIR}/model.pt"
    echo "   Logs: ${OUTPUT_BASE_DIR}/"
    echo "     - holdout_${MODEL_TYPE}_${RUN_ID}.out (stdout)"
    echo "     - holdout_${MODEL_TYPE}_${RUN_ID}.err (stderr)"
    echo "     - holdout_workflow_${MODEL_TYPE}_${RUN_ID}.log (workflow log)"
else
    echo "❌ Combined training failed"
    echo "   Logs: ${OUTPUT_BASE_DIR}/"
fi

echo ""
echo "All outputs saved to: $OUTPUT_BASE_DIR/"
echo "========================================="

exit $overall_exit_code
