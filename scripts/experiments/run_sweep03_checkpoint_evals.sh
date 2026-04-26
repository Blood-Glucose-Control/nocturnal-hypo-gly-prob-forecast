#!/usr/bin/env bash
# Evaluate every periodic checkpoint produced by 03_250k_checkpoints.yaml
# across aleppo_2017, brown_2019, and lynch_2022.
#
# The 250k training run saves snapshots/step_N/model.pt/ alongside the
# final model.pt.  This script discovers those dirs automatically and runs
# nocturnal_hypo_eval.py on each (step × dataset), producing one result
# directory per combination.
#
# USAGE
# -----
#   # Evaluate all datasets (default):
#   bash scripts/experiments/run_sweep03_checkpoint_evals.sh \
#       trained_models/artifacts/chronos2/<run_id>_holdout_workflow
#
#   # Evaluate a single dataset:
#   DATASETS="brown_2019" \
#   bash scripts/experiments/run_sweep03_checkpoint_evals.sh \
#       trained_models/artifacts/chronos2/<run_id>_holdout_workflow
#
#   # Override CUDA device (default: 0):
#   CUDA_DEVICE=1 bash scripts/experiments/run_sweep03_checkpoint_evals.sh \
#       trained_models/artifacts/chronos2/<run_id>_holdout_workflow
#
# OUTPUT
# ------
#   experiments/nocturnal_forecasting/512ctx_96fh/chronos2/
#     250k_checkpoints/
#       step_5000_aleppo_2017/
#         nocturnal_results.json
#         ...
#       step_5000_brown_2019/
#       ...
# -----------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments / defaults
# ---------------------------------------------------------------------------
ARTIFACT_DIR="${1:?Usage: $0 <artifact_dir>}"
DATASETS="${DATASETS:-aleppo_2017 brown_2019 lynch_2022}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/chronos2/99_250k_checkpoints.yaml}"

EVAL="python scripts/experiments/nocturnal_hypo_eval.py"
CONFIG_DIR="configs/data/holdout_10pct"
OUTPUT_BASE="experiments/nocturnal_forecasting/512ctx_96fh/chronos2/250k_checkpoints"

SNAPSHOTS_DIR="${ARTIFACT_DIR}/snapshots"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${ARTIFACT_DIR}" ]]; then
    echo "ERROR: artifact directory not found: ${ARTIFACT_DIR}" >&2
    exit 1
fi

if [[ ! -d "${SNAPSHOTS_DIR}" ]]; then
    echo "ERROR: no snapshots/ dir found under ${ARTIFACT_DIR}" >&2
    echo "       Has the 03_250k_checkpoints.yaml training run finished?" >&2
    exit 1
fi

# Collect and sort snapshots by step number
mapfile -t SNAPSHOT_DIRS < <(
    find "${SNAPSHOTS_DIR}" -maxdepth 2 -name "model.pt" -type d \
    | sort -t_ -k2 -n
)

if [[ ${#SNAPSHOT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no step_N/model.pt dirs found under ${SNAPSHOTS_DIR}" >&2
    exit 1
fi

N_STEPS=${#SNAPSHOT_DIRS[@]}
N_DATASETS=$(echo "${DATASETS}" | wc -w)
TOTAL=$(( N_STEPS * N_DATASETS ))

echo "=== sweep-03 checkpoint evals ==="
echo "  artifact dir : ${ARTIFACT_DIR}"
echo "  snapshots    : ${N_STEPS}  ($(basename "$(dirname "${SNAPSHOT_DIRS[0]}")")  →  $(basename "$(dirname "${SNAPSHOT_DIRS[-1]}")"))"
echo "  datasets     : ${DATASETS}"
echo "  total eval   : ${TOTAL}"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
IDX=0
for SNAPSHOT_MODEL_PT in "${SNAPSHOT_DIRS[@]}"; do
    STEP_DIR="$(dirname "${SNAPSHOT_MODEL_PT}")"     # .../snapshots/step_5000
    STEP_NAME="$(basename "${STEP_DIR}")"             # step_5000

    for DATASET in ${DATASETS}; do
        IDX=$(( IDX + 1 ))
        OUTPUT_DIR="${OUTPUT_BASE}/${STEP_NAME}_${DATASET}"

        echo "[${IDX}/${TOTAL}] ${STEP_NAME}  →  ${DATASET}"
        ${EVAL} \
            --model chronos2 \
            --dataset "${DATASET}" \
            --config-dir "${CONFIG_DIR}" \
            --model-config "${MODEL_CONFIG}" \
            --checkpoint "${SNAPSHOT_MODEL_PT}" \
            --context-length 512 \
            --forecast-length 96 \
            --cuda-device "${CUDA_DEVICE}" \
            --output-dir "${OUTPUT_DIR}" \
            --probabilistic
        echo ""
    done
done

echo "=== All checkpoint evals complete. Results in ${OUTPUT_BASE}/ ==="
