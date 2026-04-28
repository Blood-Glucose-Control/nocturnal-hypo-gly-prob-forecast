#!/usr/bin/env bash
# chronos2_long_run_step_sweep_eval.sh
#
# Evaluates the two long-run (100k step) Chronos-2 IOB checkpoint sweeps:
#   16_bg_iob_ia_high_lr_100k         (config 16, no ensemble)
#   17_bg_iob_ia_high_lr_100k_ensemble (config 17, ensemble)
#
# Every 10k snapshot (step_10000 … step_100000) plus zero-shot (step 0) is
# evaluated across aleppo_2017, brown_2019, lynch_2022 using the fixed
# episode set anchored at episode_context_length=512 (same methodology as
# the existing step-sweep and ctx-ablation experiments).
#
# ARTIFACT PATHS
# ---------------
# Both paths are hardcoded here because configs 16/17 predate the manifest
# used by the standard sweep harness.  Override with:
#   ARTIFACT_16=... ARTIFACT_17=... bash .../chronos2_long_run_step_sweep_eval.sh
#
# OUTPUT
# ---------------
# Results land in:
#   experiments/nocturnal_forecasting_long_run_step_sweep/512ctx_96fh/chronos2/
#     cfg16_step_10000_aleppo_2017/
#       results_summary.json ...
#     cfg16_step_10000_brown_2019/
#     ...
#     cfg17_step_10000_aleppo_2017/
#     ...
#
# USAGE
# -----
#   # Both GPUs (default):
#   GPUS="0 1" bash scripts/experiments/chronos2_long_run_step_sweep_eval.sh
#
#   # Single GPU:
#   GPUS="0" bash scripts/experiments/chronos2_long_run_step_sweep_eval.sh
#
#   # Override artifact dirs if they change:
#   ARTIFACT_16=trained_models/artifacts/chronos2/<id16> \
#   ARTIFACT_17=trained_models/artifacts/chronos2/<id17> \
#   bash scripts/experiments/chronos2_long_run_step_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
EVAL_SCRIPT="scripts/experiments/nocturnal_hypo_eval_ctx_ablation.py"
CONFIG_DIR="configs/data/holdout_10pct"
DATASETS=(aleppo_2017 brown_2019 lynch_2022)

OUTPUT_BASE="experiments/nocturnal_forecasting_long_run_step_sweep/512ctx_96fh/chronos2"
LOG_DIR="logs"
DONE_FILE="${LOG_DIR}/chronos2_long_run_step_sweep_done.log"

EPISODE_CTX=512

# Every 10k steps from 10k to 100k (60 jobs total; re-add 5k-steps later if needed)
STEPS=(10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

# Default artifact dirs — both long-run jobs from April 26 2026
ARTIFACT_16="${ARTIFACT_16:-trained_models/artifacts/chronos2/2026-04-26_06:26_RID20260426_062650_516757_holdout_workflow}"
ARTIFACT_17="${ARTIFACT_17:-trained_models/artifacts/chronos2/2026-04-26_06:27_RID20260426_062756_517224_holdout_workflow}"

# fmt: off
# (stem, artifact_dir, model_config)
CONFIGS=(
    "16_bg_iob_ia_high_lr_100k|${ARTIFACT_16}|configs/models/chronos2/16_bg_iob_ia_high_lr_100k.yaml"
    "17_bg_iob_ia_high_lr_100k_ensemble|${ARTIFACT_17}|configs/models/chronos2/17_bg_iob_ia_high_lr_100k_ensemble.yaml"
)
# fmt: on

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
    read -ra GPUS_ARRAY <<< "$GPUS"
else
    mapfile -t GPUS_ARRAY < <(
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo "0"
    )
fi

N_GPUS=${#GPUS_ARRAY[@]}
echo "=== Chronos-2 long-run step-sweep eval  $(date) ==="
echo "  GPUs          : ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo "  Steps         : ${STEPS[*]}"
echo "  Episode ctx   : ${EPISODE_CTX}"
echo "  Output base   : ${OUTPUT_BASE}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: chronos2 venv not found at $PYTHON" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list
# Format: "cfg_stem|artifact_dir|model_config|dataset|step"
# ---------------------------------------------------------------------------
JOBS=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r stem artifact_dir model_config <<< "$entry"

    if [[ ! -d "${PROJECT_ROOT}/${artifact_dir}" ]]; then
        echo "[WARN] artifact dir not found, skipping ${stem}: ${artifact_dir}" >&2
        continue
    fi

    for step in "${STEPS[@]}"; do
        checkpoint="${PROJECT_ROOT}/${artifact_dir}/snapshots/step_${step}/model.pt"
        if [[ ! -d "$checkpoint" ]]; then
            echo "[WARN] ${stem} step_${step} — snapshot missing: $checkpoint" >&2
            continue
        fi

        for dataset in "${DATASETS[@]}"; do
            JOBS+=("${stem}|${artifact_dir}|${model_config}|${dataset}|${checkpoint}|${step}")
        done
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Split jobs round-robin across GPUs (one worker per GPU for long-run eval)
# ---------------------------------------------------------------------------
N_SLOTS="${N_GPUS}"

declare -A SLOT_JOBS
declare -A SLOT_GPU
for (( slot=0; slot<N_SLOTS; slot++ )); do
    SLOT_JOBS[$slot]=""
    SLOT_GPU[$slot]="${GPUS_ARRAY[$slot]}"
done

for i in "${!JOBS[@]}"; do
    slot=$(( i % N_SLOTS ))
    SLOT_JOBS[$slot]+="${JOBS[$i]}"$'\n'
done

echo "Job distribution:"
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    n=$(echo "${SLOT_JOBS[$slot]}" | grep -c '|' || true)
    echo "  GPU ${gpu} worker ${slot}: ${n} jobs"
done
echo ""

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
run_gpu_worker() {
    local gpu="$1"
    local slot="$2"
    local jobs_block="$3"
    local label="GPU${gpu}w${slot}"
    local pass=0 fail=0
    local failed=()

    echo "[${label}] Starting at $(date)"

    while IFS= read -r job; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r stem artifact_dir model_config dataset checkpoint step <<< "$job"

        local done_key="${stem}|step${step}|${dataset}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already done"
            pass=$(( pass + 1 ))
            continue
        fi

        local timestamp
        timestamp="$(date +%Y-%m-%d_%H%M%S)"
        local out_dir="${PROJECT_ROOT}/${OUTPUT_BASE}/${stem}_step_${step}_${dataset}"

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  ${stem} / step_${step} / ${dataset}"
        echo "[${label}]  Checkpoint: ${checkpoint}"
        echo "[${label}]  Output:     ${out_dir}"
        echo "[${label}] ============================================"

        CMD=(
            "$PYTHON" "$EVAL_SCRIPT"
            --model chronos2
            --model-config "$model_config"
            --dataset "$dataset"
            --config-dir "$CONFIG_DIR"
            --checkpoint "$checkpoint"
            --context-length 512
            --episode-context-length "$EPISODE_CTX"
            --forecast-length 96
            --cuda-device "$gpu"
            --probabilistic
            --output-dir "$out_dir"
        )

        if CUDA_VISIBLE_DEVICES="$gpu" "${CMD[@]}"; then
            echo "[${label}] [OK] ${stem} / step_${step} / ${dataset}"
            echo "$done_key" >> "$DONE_FILE"
            pass=$(( pass + 1 ))
        else
            echo "[${label}] [FAIL] ${stem} / step_${step} / ${dataset}"
            fail=$(( fail + 1 ))
            failed+=("${stem} / step_${step} / ${dataset}")
        fi
    done <<< "$jobs_block"

    echo ""
    echo "[${label}] Done at $(date)  —  passed: ${pass} / $(( pass + fail ))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "[${label}] Failed:"
        for f in "${failed[@]}"; do
            echo "[${label}]   - $f"
        done
        return 1
    fi
}

export -f run_gpu_worker
export PYTHON EVAL_SCRIPT CONFIG_DIR DONE_FILE EPISODE_CTX PROJECT_ROOT OUTPUT_BASE

# ---------------------------------------------------------------------------
# Launch workers
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/chronos2_long_run_step_sweep_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_JOBS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched.  Waiting for completion..."
echo "(tail -f logs/chronos2_long_run_step_sweep_gpu<N>_w<slot>.log to monitor)"
echo ""

OVERALL_FAIL=0
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    pid="${PIDS[$slot]}"
    if wait "$pid"; then
        echo "GPU ${gpu} worker ${slot}: SUCCESS"
    else
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/chronos2_long_run_step_sweep_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Chronos-2 long-run step-sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures — check logs above."
    exit 1
fi
echo "  All jobs passed."
