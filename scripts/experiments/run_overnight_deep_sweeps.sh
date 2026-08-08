#!/usr/bin/env bash
# run_overnight_deep_sweeps.sh
#
# Sequentially runs the DeepAR -> PatchTST -> TFT sweeps end-to-end.
# Each stage is independent: a failure in one stage does NOT abort the chain.
# After every stage we diff the model's sweep_manifest.txt against the YAMLs
# under configs/models/<model>/ and append any missing stems to a master
# retry manifest so we can re-launch them tomorrow.
#
# Outputs:
#   logs/overnight_chain.log                — top-level orchestrator log
#   logs/overnight_<model>_train.log        — per-stage stdout/stderr
#   logs/overnight_failed_retry.txt         — TSV: <model>\t<config_path>\t<datasets_key>
#
# Usage:
#   nohup bash scripts/experiments/run_overnight_deep_sweeps.sh \
#       > logs/overnight_chain.log 2>&1 &
#
# Tail with:
#   tail -f logs/overnight_chain.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CHAIN_LOG="${LOG_DIR}/overnight_chain.log"
RETRY_FILE="${LOG_DIR}/overnight_failed_retry.txt"
: > "$RETRY_FILE"

# ---------------------------------------------------------------------------
# Parse a sweep_train.sh CONFIGS=( ... ) array into TSV: stem<TAB>path<TAB>key
# ---------------------------------------------------------------------------
parse_sweep_configs() {
    local sweep_script="$1"
    awk '
        /^CONFIGS=\(/ { in_arr=1; next }
        in_arr && /^\)/ { in_arr=0; next }
        in_arr {
            line=$0
            gsub(/^[[:space:]]*"/, "", line)
            gsub(/"[[:space:]]*$/, "", line)
            n = split(line, parts, "|")
            if (n == 2) {
                path = parts[1]
                key  = parts[2]
                m = split(path, segs, "/")
                stem = segs[m]
                sub(/\.yaml$/, "", stem)
                printf "%s\t%s\t%s\n", stem, path, key
            }
        }
    ' "$sweep_script"
}

# ---------------------------------------------------------------------------
# After a stage, append failed configs (those not in manifest) to RETRY_FILE
# ---------------------------------------------------------------------------
record_failures() {
    local model="$1"
    local sweep_script="$2"
    local manifest="trained_models/artifacts/${model}/sweep_manifest.txt"
    [[ -f "$manifest" ]] || touch "$manifest"

    local total=0 ok=0 missing=0
    while IFS=$'\t' read -r stem path key; do
        [[ -z "$stem" ]] && continue
        total=$(( total + 1 ))
        if awk -F'\t' -v s="$stem" '$1 == s { found=1 } END { exit !found }' "$manifest"; then
            ok=$(( ok + 1 ))
        else
            missing=$(( missing + 1 ))
            printf '%s\t%s\t%s\n' "$model" "$path" "$key" >> "$RETRY_FILE"
        fi
    done < <(parse_sweep_configs "$sweep_script")

    echo "[chain] ${model}: ${ok}/${total} succeeded, ${missing} need retry"
}

# ---------------------------------------------------------------------------
# Run one stage; never aborts the chain
# ---------------------------------------------------------------------------
run_stage() {
    local model="$1"
    local sweep_script="scripts/experiments/${model}_sweep_train.sh"
    local stage_log="${LOG_DIR}/overnight_${model}_train.log"

    echo ""
    echo "[chain] ============================================================"
    echo "[chain]  Stage: ${model}    started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[chain]  Script: ${sweep_script}"
    echo "[chain]  Log:    ${stage_log}"
    echo "[chain] ============================================================"

    if [[ ! -x "$sweep_script" && ! -f "$sweep_script" ]]; then
        echo "[chain] ERROR: sweep script not found: ${sweep_script}"
        return 0
    fi

    local rc=0
    bash "$sweep_script" > "$stage_log" 2>&1 || rc=$?
    echo "[chain]  Stage: ${model}    finished $(date '+%Y-%m-%d %H:%M:%S')  exit=${rc}"
    record_failures "$model" "$sweep_script"
    return 0
}

# ---------------------------------------------------------------------------
# Run the chain
# ---------------------------------------------------------------------------
echo "[chain] === Overnight deep sweep chain  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "[chain] Project: ${PROJECT_ROOT}"
echo "[chain] Retry manifest: ${RETRY_FILE}"

run_stage deepar
run_stage patchtst
run_stage tft

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
total_failures=$(grep -c '' "$RETRY_FILE" 2>/dev/null || echo 0)

echo ""
echo "[chain] ============================================================"
echo "[chain]  Chain complete  $(date '+%Y-%m-%d %H:%M:%S')"
echo "[chain]  Stages run: deepar, patchtst, tft"
echo "[chain]  Total failed configs: ${total_failures}"
echo "[chain]  Retry manifest: ${RETRY_FILE}"
if [[ "$total_failures" -gt 0 ]]; then
    echo "[chain]  --- Failed configs ---"
    cat "$RETRY_FILE"
fi
echo "[chain] ============================================================"
