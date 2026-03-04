#!/bin/bash
# =============================================================================
# SLURM: Chronos-2 Generic Holdout Workflow
# =============================================================================
# Runs the generic holdout workflow script with --model-type chronos2.
# Uses the bg_only_test.yaml config (100 steps) for a fast smoke test.
#
# Usage:
#   sbatch scripts/training/slurm/chronos2_holdout_workflow.sh
#
#   # Override datasets or steps:
#   sbatch --export=DATASETS="brown_2019",STEPS=5000 \
#     scripts/training/slurm/chronos2_holdout_workflow.sh
#
#   # Skip training (zero-shot only):
#   sbatch --export=SKIP_TRAINING=1 \
#     scripts/training/slurm/chronos2_holdout_workflow.sh
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

# ── Environment setup ───────────────────────────────────────────────────────
PROJECT_ROOT="/u201/s6jindal/nocturnal-hypo-gly-prob-forecast"
cd "$PROJECT_ROOT"

# Activate conda environment
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

# GPU memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Create log directory
mkdir -p slurm_logs

# ── Print job info ──────────────────────────────────────────────────────────
echo "============================================================"
echo "Chronos-2 Holdout Workflow"
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
CMD="python scripts/examples/example_holdout_generic_workflow.py"
CMD+=" --model-type chronos2"
CMD+=" --datasets $DATASETS"
CMD+=" --config-dir $CONFIG_DIR"
CMD+=" --model-config $MODEL_CONFIG"

if [ "$SKIP_TRAINING" = "1" ] || [ "$SKIP_TRAINING" = "true" ]; then
    CMD+=" --skip-training"
fi

if [ -n "$SKIP_STEPS" ]; then
    CMD+=" --skip-steps $SKIP_STEPS"
fi

if [ -n "$EPOCHS" ]; then
    CMD+=" --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD+=" --batch-size $BATCH_SIZE"
fi

# ── Run ─────────────────────────────────────────────────────────────────────
echo ""
echo ">>> $CMD"
echo ""
eval "$CMD"
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
