#!/usr/bin/env bash
# statistical_sweep_eval.sh
#
# Evaluates all statistical baseline models from the trained manifest.
# CPU-only. Uses JOBS_PER_CPU parallel workers (default: nproc / 2).
#
# Checkpoint paths are read from the manifest written by statistical_sweep_train.sh:
#   trained_models/artifacts/statistical/sweep_manifest.txt
# Format: <stem>\t<output_dir>  (tab-separated, last entry wins on re-run)
#
# Usage:
#   bash scripts/experiments/statistical_sweep_eval.sh
#   JOBS_PER_CPU=4 bash scripts/experiments/statistical_sweep_eval.sh 2>&1 | tee statistical_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
MANIFEST="trained_models/artifacts/statistical/sweep_manifest.txt"
LOG_DIR="logs"

echo "=== Statistical sweep eval  $(date) ==="
echo "  Venv: ${PYTHON}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run statistical_sweep_train.sh first."
    exit 1
fi

# Format: "stem|context_length|space-separated covariate cols|datasets_key"
CONFIG_META=(
    "00_autoarima_bg_only|512||ALL"
    "01_autoarima_bg_iob|512|iob|IOB"
    "02_theta_bg_only|512||ALL"
    "03_npts_bg_only|512||ALL"
)

mkdir -p "$LOG_DIR"
DONE_FILE="${LOG_DIR}/statistical_eval_done.log"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list: "stem|ctx_len|cov_cols|dataset|checkpoint"
# ---------------------------------------------------------------------------
JOBS=()
for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"

    if [[ "$datasets_key" == "IOB" ]]; then
        ft_datasets=("${DATASETS_WITH_IOB[@]}")
    else
        ft_datasets=("${ALL_DATASETS[@]}")
    fi

    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"
    checkpoint=""
    if [[ -n "$out_dir" ]]; then
        checkpoint="${PROJECT_ROOT}/${out_dir}"
    fi

    for dataset in "${ft_datasets[@]}"; do
        if [[ -z "$out_dir" ]]; then
            echo "[WARN] ${stem} — not in manifest, jobs will be skipped"
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        elif [[ ! -d "$checkpoint" ]]; then
            echo "[WARN] ${stem} — output dir missing: ${checkpoint}"
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        else
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|${checkpoint}")
        fi
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Parallelism (CPU-only)
# ---------------------------------------------------------------------------
N_CPUS=$(nproc 2>/dev/null || echo 4)
JOBS_PER_CPU="${JOBS_PER_CPU:-$(( N_CPUS / 2 ))}"
[[ $JOBS_PER_CPU -lt 1 ]] && JOBS_PER_CPU=1
echo "  CPUs available: ${N_CPUS},  parallel workers: ${JOBS_PER_CPU}"
echo ""

declare -A WORKER_JOBS
for (( w=0; w<JOBS_PER_CPU; w++ )); do
    WORKER_JOBS[$w]=""
done
for i in "${!JOBS[@]}"; do
    w=$(( i % JOBS_PER_CPU ))
    WORKER_JOBS[$w]+="${JOBS[$i]}"$'\n'
done

# ---------------------------------------------------------------------------
# Worker function
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
        IFS='|' read -r stem ctx_len cov_cols dataset checkpoint <<< "$job"

        local model_config="configs/models/statistical/${stem}.yaml"
        local data_config="${CONFIG_DIR}/${dataset}.yaml"
        local done_key="${stem}|${dataset}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already done"
            pass=$(( pass + 1 ))
            continue
        fi

        if [[ "$checkpoint" == "MISSING" ]]; then
            echo "[${label}] [SKIP] ${done_key} — checkpoint missing"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi

        if [[ ! -f "$data_config" ]]; then
            echo "[${label}] [SKIP] ${done_key} — data config not found: ${data_config}"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  Eval: statistical / ${stem} / ${dataset}"
        echo "[${label}]  Checkpoint: ${checkpoint}"
        echo "[${label}] ============================================"

        eval_args=(
            --probabilistic
            --model-config "$model_config"
            --data-config "$data_config"
            --checkpoint "$checkpoint"
        )
        if [[ -n "$cov_cols" ]]; then
            # shellcheck disable=SC2206
            eval_args+=(--covariate-cols $cov_cols)
        fi

        if "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py "${eval_args[@]}"; then
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
export PYTHON CONFIG_DIR DONE_FILE

declare -A PIDS
for (( w=0; w<JOBS_PER_CPU; w++ )); do
    log_file="${LOG_DIR}/statistical_eval_w${w}.log"
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
        echo "Worker ${w}: FAILED  (see ${LOG_DIR}/statistical_eval_w${w}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Statistical sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} worker(s) reported failures."
    exit 1
fi
echo "  All configs evaluated successfully."
