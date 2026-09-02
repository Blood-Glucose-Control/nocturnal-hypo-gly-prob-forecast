#!/usr/bin/env bash
#
# PRODUCTION SINGLE-GPU WORKFLOW LAUNCHER
# =======================================
# Rewired to the maintained forecasting workflow wrapper:
#   scripts/workflows/forecasting/run_forecasting_workflow.sh
#
# Quick start:
#   sbatch scripts/training/slurm/single_gpu.sh
#
# Backward-compatible override:
#   sbatch --export=CONFIG_PATH=configs/models/ttm/custom.yaml scripts/training/slurm/single_gpu.sh
#
# Preferred overrides:
#   sbatch --export=MODEL_TYPE=chronos2,MODEL_CONFIG=configs/models/chronos2/00_bg_only.yaml,DATASETS="brown_2019 lynch_2022"
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

set -euo pipefail

datasets_was_set="${DATASETS+x}"
config_dir_was_set="${CONFIG_DIR+x}"

# Backward-compatible names
: "${CONFIG_PATH:=configs/models/ttm/fine_tune.yaml}"
: "${DATA_CONFIG:=}"
: "${OUTPUT_DIR:=trained_models/artifacts/ttm}"
: "${EXPERIMENT_NAME:=single_gpu_training}"

# Canonical wrapper inputs
: "${MODEL_TYPE:=ttm}"
: "${MODEL_CONFIG:=$CONFIG_PATH}"
: "${DATASETS:=brown_2019}"
: "${CONFIG_DIR:=configs/data/holdout_10pct}"
: "${OUTPUT_BASE_DIR:=${OUTPUT_DIR%/}/${EXPERIMENT_NAME}}"
: "${SKIP_TRAINING:=false}"
: "${SKIP_STEPS:=1 2 4 6 7}"
: "${EPOCHS:=}"
: "${BATCH_SIZE:=}"
: "${VENV_NAME:=$MODEL_TYPE}"
: "${DRY_RUN:=0}"
: "${RUN_ID:=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}}"

if [[ -n "$DATA_CONFIG" ]]; then
    if [[ -z "$datasets_was_set" ]]; then
        DATASETS="$(basename "${DATA_CONFIG%.yaml}")"
    fi
    if [[ -z "$config_dir_was_set" ]]; then
        CONFIG_DIR="$(dirname "$DATA_CONFIG")"
    fi
fi

case "$SKIP_TRAINING" in
    1 | true | TRUE | yes | YES)
        SKIP_TRAINING="true"
        ;;
    0 | false | FALSE | no | NO | "")
        SKIP_TRAINING="false"
        ;;
    *)
        echo "ERROR: SKIP_TRAINING must be one of: true/false/1/0/yes/no"
        exit 1
        ;;
esac

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

WORKFLOW_SCRIPT="scripts/workflows/forecasting/run_forecasting_workflow.sh"
if [[ ! -f "$WORKFLOW_SCRIPT" ]]; then
    echo "ERROR: workflow script not found: $WORKFLOW_SCRIPT"
    exit 1
fi

echo "========================================="
echo "Single-GPU forecasting workflow launcher"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Project root: $PROJECT_ROOT"
echo "Model type: $MODEL_TYPE"
echo "Model config: $MODEL_CONFIG"
echo "Datasets: $DATASETS"
echo "Config dir: $CONFIG_DIR"
echo "Output base dir: $OUTPUT_BASE_DIR"
echo "Skip training: $SKIP_TRAINING"
echo "Skip steps: ${SKIP_STEPS:-none}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Run ID: $RUN_ID"
echo "========================================="
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 set; command path validated without executing workflow."
    exit 0
fi

MODEL_TYPE="$MODEL_TYPE" \
MODEL_CONFIG="$MODEL_CONFIG" \
DATASETS="$DATASETS" \
CONFIG_DIR="$CONFIG_DIR" \
OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR" \
SKIP_TRAINING="$SKIP_TRAINING" \
SKIP_STEPS="$SKIP_STEPS" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
VENV_NAME="$VENV_NAME" \
RUN_ID="$RUN_ID" \
bash "$WORKFLOW_SCRIPT"
exit_code=$?

echo ""
echo "========================================="
echo "Workflow completed: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit "$exit_code"
