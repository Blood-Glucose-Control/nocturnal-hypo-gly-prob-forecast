#!/usr/bin/env bash
# naive_baseline_sweep_eval.sh
#
# Evaluates Naive and Average baseline models across all 4 datasets.
# Reads trained checkpoint paths from the manifest written by
# naive_baseline_sweep_train.sh:
#   trained_models/artifacts/naive_baseline/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, last entry per stem wins)
#
# NOTE: PIT histograms for these models will show miscalibration. This is
# intentional — synthetic quantiles from residuals are reported as a contrast
# to natively probabilistic models (NPTS, DeepAR, PatchTST, TFT, Chronos-2).
#
# CPU-only. No GPU required.
# Use JOBS_PER_CPU to parallelise across datasets (default: nproc / 2).
#
# Usage:
#   bash scripts/experiments/naive_baseline_sweep_eval.sh
#   JOBS_PER_CPU=4 bash scripts/experiments/naive_baseline_sweep_eval.sh 2>&1 | tee logs/naive_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
MANIFEST="trained_models/artifacts/naive_baseline/sweep_manifest.txt"
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Smoke-test import: confirm AutoGluon Naive/Average are available
# ---------------------------------------------------------------------------
echo "=== Naive Baseline sweep eval  $(date) ==="
echo "  Venv: ${PYTHON}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

"$PYTHON" - <<'EOF'
from autogluon.timeseries import TimeSeriesPredictor
print("AutoGluon import OK")
EOF
echo ""

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run naive_baseline_sweep_train.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Config definitions — "stem|context_length"
# All naive/average configs use the full dataset set (BG-only, no covariates)
# ---------------------------------------------------------------------------
CONFIGS=(
    "00_naive|512"
    "01_average|512"
)

# ---------------------------------------------------------------------------
# Parallelism: JOBS_PER_CPU workers (no GPU needed)
# ---------------------------------------------------------------------------
N_CPUS=$(nproc 2>/dev/null || echo 4)
JOBS_PER_CPU="${JOBS_PER_CPU:-$(( N_CPUS / 2 ))}"
[[ $JOBS_PER_CPU -lt 1 ]] && JOBS_PER_CPU=1
echo "  CPUs available: ${N_CPUS},  parallel workers: ${JOBS_PER_CPU}"
echo ""

mkdir -p "$LOG_DIR"
DONE_FILE="${LOG_DIR}/naive_baseline_eval_done.log"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list: "stem|ctx_len|dataset"
# ---------------------------------------------------------------------------
JOBS=()
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r stem ctx_len <<< "$entry"
    for dataset in "${ALL_DATASETS[@]}"; do
        JOBS+=("${stem}|${ctx_len}|${dataset}")
    done
done
echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Distribute jobs across workers
# ---------------------------------------------------------------------------
declare -A WORKER_JOBS
for (( w=0; w<JOBS_PER_CPU; w++ )); do
    WORKER_JOBS[$w]=""
done
for i in "${!JOBS[@]}"; do
    w=$(( i % JOBS_PER_CPU ))
    WORKER_JOBS[$w]+="${JOBS[$i]}"$'\n'
done

# ---------------------------------------------------------------------------
# Worker function (CPU-only)
# ---------------------------------------------------------------------------
run_cpu_worker() {
    local worker="$1"
    local jobs_block="$2"
    local label="CPU_w${worker}"
    local pass=0 fail=0
    local failed=()

    echo "[${label}] Starting at $(date)"

    while IFS= read -r job; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r stem ctx_len dataset <<< "$job"

        local model_config="configs/models/naive_baseline/${stem}.yaml"
        local done_key="${stem}|${dataset}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already done"
            pass=$(( pass + 1 ))
            continue
        fi

        # Look up checkpoint from manifest (last match wins)
        local out_dir
        out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"
        if [[ -z "$out_dir" ]]; then
            echo "[${label}] [SKIP] ${done_key} — not in manifest, run naive_baseline_sweep_train.sh first"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi
        local checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"
        if [[ ! -d "$checkpoint" ]]; then
            echo "[${label}] [SKIP] ${done_key} — checkpoint not found: $checkpoint"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  Eval: naive_baseline / ${stem} / ${dataset}"
        echo "[${label}]  Checkpoint: ${checkpoint}"
        echo "[${label}] ============================================"

        if "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py \
               --model naive_baseline \
               --model-config "$model_config" \
               --dataset "$dataset" \
               --config-dir "$CONFIG_DIR" \
               --checkpoint "$checkpoint" \
               --context-length "$ctx_len" \
               --forecast-length 96 \
               --probabilistic; then
            echo "[${label}] [OK] ${done_key}"
            echo "$done_key" >> "$DONE_FILE"
            pass=$(( pass + 1 ))
        else
            echo "[${label}] [FAIL] ${done_key}"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
        fi
    done <<< "$jobs_block"

    echo ""
    echo "[${label}] Done at $(date)  —  passed: ${pass} / $(( pass + fail ))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "[${label}] Failed: ${failed[*]}"
        return 1
    fi
}

export -f run_cpu_worker
export PYTHON CONFIG_DIR DONE_FILE MANIFEST PROJECT_ROOT

# ---------------------------------------------------------------------------
# Launch workers in background
# ---------------------------------------------------------------------------
declare -A PIDS
for (( w=0; w<JOBS_PER_CPU; w++ )); do
    log_file="${LOG_DIR}/naive_baseline_eval_w${w}.log"
    echo "Launching worker ${w} → ${log_file}"
    run_cpu_worker "$w" "${WORKER_JOBS[$w]}" > "$log_file" 2>&1 &
    PIDS[$w]=$!
done

echo ""
echo "All workers launched. Waiting..."
echo ""

OVERALL_FAIL=0
for (( w=0; w<JOBS_PER_CPU; w++ )); do
    if wait "${PIDS[$w]}"; then
        echo "Worker ${w}: SUCCESS"
    else
        echo "Worker ${w}: FAILED  (see ${LOG_DIR}/naive_baseline_eval_w${w}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Naive Baseline sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} worker(s) reported failures — check logs."
    exit 1
fi
echo "  All configs evaluated successfully."
