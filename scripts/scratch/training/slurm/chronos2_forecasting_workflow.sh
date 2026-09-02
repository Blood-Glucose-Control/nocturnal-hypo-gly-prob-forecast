#!/bin/bash
# =============================================================================
# SLURM: Chronos-2 Generic Forecasting Workflow
# =============================================================================
# Runs the generic forecasting workflow script with --model-type chronos2.
# Uses the bg_only_test.yaml config (100 steps) for a fast smoke test.
#
# Usage:
#   sbatch scripts/training/slurm/chronos2_forecasting_workflow.sh
#
#   # Override datasets or steps:
#   sbatch --export=DATASETS="brown_2019",EPOCHS=1 \
#     scripts/training/slurm/chronos2_forecasting_workflow.sh
#
#   # Skip training (zero-shot only):
#   sbatch --export=SKIP_TRAINING=1 \
#     scripts/training/slurm/chronos2_forecasting_workflow.sh
# =============================================================================

#SBATCH --job-name=c2_workflow
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/chronos2_workflow_%j.log

set -euo pipefail

# ── Configurable variables (override via sbatch --export=...) ────────────────
: ${DATASETS:="brown_2019"}
: ${CONFIG_DIR:="configs/data/holdout_10pct"}
: ${MODEL_CONFIG:="configs/models/chronos2/bg_only_test.yaml"}
: ${SKIP_TRAINING:=0}
: ${SKIP_STEPS:=""}
: ${EPOCHS:=""}
: ${BATCH_SIZE:=""}
: ${DRY_RUN:=0}

# ── Environment setup ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment for real runs.
if [ "$DRY_RUN" != "1" ]; then
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
    conda activate chronos
fi

# GPU memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMpatient_id_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Create log directory
mkdir -p slurm_logs

# ── Print job info ──────────────────────────────────────────────────────────
echo "============================================================"
echo "Chronos-2 Forecasting Workflow"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Datasets:     $DATASETS"
echo "Config dir:   $CONFIG_DIR"
echo "Model config: $MODEL_CONFIG"
echo "Skip train:   $SKIP_TRAINING"
echo "Python:       $(which python)"
echo "============================================================"

# ── Build command ───────────────────────────────────────────────────────────
read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a SKIP_STEP_ARGS <<< "$SKIP_STEPS"

CMD=(python scripts/workflows/forecasting/forecasting_workflow_orchestrator.py)
CMD+=(--model-type chronos2)
CMD+=(--datasets "${DATASET_ARGS[@]}")
CMD+=(--config-dir "$CONFIG_DIR")
CMD+=(--model-config "$MODEL_CONFIG")

if [ "$SKIP_TRAINING" = "1" ] || [ "$SKIP_TRAINING" = "true" ]; then
    CMD+=(--skip-training)
fi

if [ -n "$SKIP_STEPS" ]; then
    CMD+=(--skip-steps "${SKIP_STEP_ARGS[@]}")
fi

if [ -n "$EPOCHS" ]; then
    CMD+=(--epochs "$EPOCHS")
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch-size "$BATCH_SIZE")
fi

# ── Run ─────────────────────────────────────────────────────────────────────
echo ""
printf ">>> "
printf "%q " "${CMD[@]}"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 set; command path validated without executing workflow."
    exit 0
fi

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
