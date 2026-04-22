#!/usr/bin/env bash
# moment_sweep_eval.sh
#
# Evaluates all MOMENT sweep checkpoints in two parallel lanes:
#   - Lane A runs sequentially on GPU 0
#   - Lane B runs sequentially on GPU 1
#
# Configs are discovered from configs/models/moment/[0-9][0-9]_*.yaml,
# sorted, and split alternately across the two lanes.
#
# Checkpoints are resolved from:
#   trained_models/artifacts/moment/sweep_manifest.txt
# Format:
#   <stem>\t<output_dir>
#
# Dataset filtering is covariate-aware per config:
#   - BG-only: lynch, aleppo, brown, tamborlane
#   - IOB/insulin covariates: lynch, aleppo, brown
#   - COB/carb covariates: lynch, aleppo
#
# Usage:
#   bash scripts/experiments/moment_sweep_eval.sh
#   GPU0=0 GPU1=1 bash scripts/experiments/moment_sweep_eval.sh
#   DATASETS="aleppo_2017" bash scripts/experiments/moment_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/moment/bin/python"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
REQUESTED_DATASETS_STR="${DATASETS:-lynch_2022 aleppo_2017 brown_2019 tamborlane_2008}"
CONFIG_DIR="${CONFIG_DIR:-configs/data/holdout_10pct}"
MANIFEST="trained_models/artifacts/moment/sweep_manifest.txt"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: moment venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run moment_sweep_train.sh first."
    exit 1
fi

mapfile -t ALL_CONFIGS < <(ls configs/models/moment/[0-9][0-9]_*.yaml 2>/dev/null | sort)
if [[ ${#ALL_CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: no MOMENT sweep configs found at configs/models/moment/[0-9][0-9]_*.yaml"
    exit 1
fi

CONFIGS_GPU0=()
CONFIGS_GPU1=()
for i in "${!ALL_CONFIGS[@]}"; do
    if (( i % 2 == 0 )); then
        CONFIGS_GPU0+=("${ALL_CONFIGS[$i]}")
    else
        CONFIGS_GPU1+=("${ALL_CONFIGS[$i]}")
    fi
done

IFS=' ' read -r -a REQUESTED_DATASETS <<< "$REQUESTED_DATASETS_STR"
if [[ ${#REQUESTED_DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: DATASETS is empty"
    exit 1
fi

get_config_covariates() {
    local config="$1"
    "$PYTHON" - "$config" << 'PYEOF' 2>/dev/null || true
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print("")
    raise SystemExit(0)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
covs = cfg.get("covariate_cols") or []
if isinstance(covs, str):
    covs = [covs]
covs = [str(c).strip() for c in covs if str(c).strip()]
print(" ".join(covs))
PYEOF
}

resolve_datasets_for_covariates() {
    local covs_str="$1"
    local allowed=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)

    if [[ "$covs_str" == *"cob"* || "$covs_str" == *"carb_availability"* ]]; then
        allowed=(lynch_2022 aleppo_2017)
    elif [[ "$covs_str" == *"iob"* || "$covs_str" == *"insulin_availability"* ]]; then
        allowed=(lynch_2022 aleppo_2017 brown_2019)
    fi

    local resolved=()
    for ds in "${REQUESTED_DATASETS[@]}"; do
        for a in "${allowed[@]}"; do
            if [[ "$ds" == "$a" ]]; then
                resolved+=("$ds")
                break
            fi
        done
    done

    echo "${resolved[*]}"
}

run_eval_lane() {
    local lane_name="$1"
    local gpu="$2"
    shift 2
    local configs=("$@")

    local pass=0
    local fail=0
    local failed=()

    echo ""
    echo "============================================================"
    echo "  Starting lane: ${lane_name} (GPU ${gpu})"
    echo "  Config count: ${#configs[@]}"
    echo "============================================================"

    for config in "${configs[@]}"; do
        local stem
        stem="$(basename "$config" .yaml)"
        local covs
        covs="$(get_config_covariates "$config")"
        local resolved_datasets
        resolved_datasets="$(resolve_datasets_for_covariates "$covs")"
        if [[ -z "$resolved_datasets" ]]; then
            echo "[SKIP] ${lane_name}: moment / ${stem} (no compatible datasets after covariate filtering)"
            fail=$((fail + 1))
            failed+=("moment / ${stem} (no compatible datasets)")
            continue
        fi
        local cov_array=()
        if [[ -n "$covs" ]]; then
            IFS=' ' read -r -a cov_array <<< "$covs"
        fi

        local ctx_len
        ctx_len="$(awk -F': ' '/^context_length:/ {print $2; exit}' "$config")"
        if [[ -z "$ctx_len" ]]; then
            echo "[FAIL] ${lane_name}: could not read context_length from ${config}"
            fail=$((fail + 1))
            failed+=("moment / ${stem} (missing context_length)")
            continue
        fi

        local out_dir
        out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"
        if [[ -z "$out_dir" ]]; then
            echo "[SKIP] ${lane_name}: moment / ${stem} not found in manifest"
            fail=$((fail + 1))
            failed+=("moment / ${stem} (missing manifest entry)")
            continue
        fi

        local checkpoint
        checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"
        if [[ ! -d "$checkpoint" ]]; then
            echo "[SKIP] ${lane_name}: moment / ${stem} checkpoint missing: ${checkpoint}"
            fail=$((fail + 1))
            failed+=("moment / ${stem} (missing checkpoint)")
            continue
        fi

        local eval_datasets=()
        IFS=' ' read -r -a eval_datasets <<< "$resolved_datasets"
        for dataset in "${eval_datasets[@]}"; do
            local label
            label="moment / ${stem} / ${dataset}"

            echo ""
            echo "------------------------------------------------------------"
            echo "  Lane: ${lane_name}"
            echo "  Eval: ${label}"
            echo "  Covariates: ${covs:-none}"
            echo "  Checkpoint: ${checkpoint}"
            echo "------------------------------------------------------------"

            CMD=(
                "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
                --model moment
                --model-config "$config"
                --dataset "$dataset"
                --config-dir "$CONFIG_DIR"
                --checkpoint "$checkpoint"
                --context-length "$ctx_len"
                --forecast-length 96
                --cuda-device "$gpu"
            )
            if [[ ${#cov_array[@]} -gt 0 ]]; then
                CMD+=(--covariate-cols "${cov_array[@]}")
            fi

            if "${CMD[@]}"; then
                echo "[OK] ${label}"
                pass=$((pass + 1))
            else
                echo "[FAIL] ${label}"
                fail=$((fail + 1))
                failed+=("${label}")
            fi
        done
    done

    echo ""
    echo "============================================================"
    echo "  Lane complete: ${lane_name}"
    echo "  Passed: ${pass} / $((pass + fail))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  Failed in ${lane_name}:"
        for f in "${failed[@]}"; do
            echo "    - ${f}"
        done
    fi
    echo "============================================================"

    [[ $fail -eq 0 ]]
}

run_eval_lane "gpu${GPU0}" "$GPU0" "${CONFIGS_GPU0[@]}" &
PID0=$!
run_eval_lane "gpu${GPU1}" "$GPU1" "${CONFIGS_GPU1[@]}" &
PID1=$!

FAIL_LANES=0
if ! wait "$PID0"; then
    FAIL_LANES=$((FAIL_LANES + 1))
fi
if ! wait "$PID1"; then
    FAIL_LANES=$((FAIL_LANES + 1))
fi

echo ""
echo "============================================================"
echo "  MOMENT sweep eval complete  $(date)"
echo "  Lanes failed: ${FAIL_LANES}"
echo "============================================================"

[[ $FAIL_LANES -eq 0 ]]
