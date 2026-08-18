#!/usr/bin/env bash
#
# ADAPTIVE WORKFLOW LAUNCHER
# ==========================
# Rewired to the maintained forecasting workflow wrapper:
#   scripts/workflows/forecasting/run_forecasting_workflow.sh
#
# Quick start:
#   sbatch scripts/training/slurm/adaptive_resources.sh
#
# Override detection:
#   sbatch --export=FORCE_NUM_GPUS=2 scripts/training/slurm/adaptive_resources.sh
#
#SBATCH --job-name=ttm_train_auto
#SBATCH --output=logs/train_auto_%j.out
#SBATCH --error=logs/train_auto_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
##SBATCH --mail-user=your.email@example.com
##SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

datasets_was_set="${DATASETS+x}"
config_dir_was_set="${CONFIG_DIR+x}"

# Backward-compatible names
: "${FORCE_NUM_GPUS:=}"
: "${CONFIG_PATH:=configs/models/ttm/fine_tune.yaml}"
: "${DATA_CONFIG:=}"
: "${OUTPUT_DIR:=trained_models/artifacts/ttm}"
: "${EXPERIMENT_NAME:=adaptive_training}"

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

detect_gpus() {
    local -a detected
    mapfile -t detected < <(
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true
    )
    if (( ${#detected[@]} > 0 )); then
        printf "%s\n" "${detected[@]}"
    fi
}

mapfile -t available_gpus < <(detect_gpus)
available_count="${#available_gpus[@]}"

if [[ -n "$FORCE_NUM_GPUS" ]]; then
    target_num_gpus="$FORCE_NUM_GPUS"
else
    target_num_gpus="$available_count"
fi

if (( target_num_gpus < 0 )); then
    echo "ERROR: FORCE_NUM_GPUS must be >= 0"
    exit 1
fi

declare -a selected_gpus
if (( target_num_gpus == 0 )) || (( available_count == 0 )); then
    strategy="cpu"
else
    if (( target_num_gpus > available_count )); then
        target_num_gpus="$available_count"
    fi
    selected_gpus=("${available_gpus[@]:0:target_num_gpus}")
    if (( target_num_gpus == 1 )); then
        strategy="single_gpu"
    else
        strategy="multi_gpu"
    fi
fi

cpus_per_task="${SLURM_CPUS_PER_TASK:-32}"
if [[ "$strategy" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cpus_per_task}"
    export WORLD_SIZE=1
else
    gpu_csv="$(IFS=,; echo "${selected_gpus[*]}")"
    threads_per_gpu=$((cpus_per_task / target_num_gpus))
    if (( threads_per_gpu < 1 )); then
        threads_per_gpu=1
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu_csv"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads_per_gpu}"
    export WORLD_SIZE="$target_num_gpus"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

WORKFLOW_SCRIPT="scripts/workflows/forecasting/run_forecasting_workflow.sh"
if [[ ! -f "$WORKFLOW_SCRIPT" ]]; then
    echo "ERROR: workflow script not found: $WORKFLOW_SCRIPT"
    exit 1
fi

echo "========================================="
echo "Adaptive forecasting workflow launcher"
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
echo "Detected GPUs: $available_count"
echo "Target GPUs: $target_num_gpus"
echo "Strategy: $strategy"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "WORLD_SIZE: $WORLD_SIZE"
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
