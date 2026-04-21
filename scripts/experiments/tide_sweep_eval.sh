#!/usr/bin/env bash
# tide_sweep_eval.sh
#
# Runs nocturnal_hypo_eval.py --probabilistic for every TiDE sweep config
# (00–06) × the appropriate dataset subset:
#   ALL (4 datasets): bg-only config (00)
#   IOB (3 datasets, excl. tamborlane): insulin covariate configs (01, 02, 04–06)
#   COB (2 datasets, excl. brown+tamborlane): carb covariate config (03)
#
# Checkpoint paths are read from the manifest written by tide_sweep_train.sh:
#   trained_models/artifacts/tide/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, one entry per config)
# If a stem appears more than once (re-run), the last entry wins.
#
# Usage:
#   bash scripts/experiments/tide_sweep_eval.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/tide_sweep_eval.sh
#   bash scripts/experiments/tide_sweep_eval.sh 2>&1 | tee tide_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/tide/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
DATASETS_WITH_COB=(lynch_2022 aleppo_2017)
MANIFEST="trained_models/artifacts/tide/sweep_manifest.txt"

# Format: "stem|context_length|space-separated covariate cols (empty = BG only)|datasets_key"
CONFIGS=(
    "00_bg_only|512||ALL"
    "01_bg_iob|512|iob|IOB"
    "02_bg_iob_insulin_availability|512|iob insulin_availability|IOB"
    "03_bg_all_covariates|512|iob insulin_availability cob|COB"
    "04_large_arch|512|iob insulin_availability|IOB"
    "05_bg_iob_ia_high_lr|512|iob insulin_availability|IOB"
    "06_bg_iob_ia_short_ctx|288|iob insulin_availability|IOB"
)

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: tide venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run tide_sweep_train.sh first."
    exit 1
fi

PASS=0
FAIL=0
FAILED=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"
    model_config="configs/models/tide/${stem}.yaml"

    if [[ "$datasets_key" == "IOB" ]]; then
        datasets=("${DATASETS_WITH_IOB[@]}")
    elif [[ "$datasets_key" == "COB" ]]; then
        datasets=("${DATASETS_WITH_COB[@]}")
    else
        datasets=("${ALL_DATASETS[@]}")
    fi

    # Look up the output dir for this stem in the manifest (last match wins,
    # so re-runs naturally use the most recent training)
    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"

    if [[ -z "$out_dir" ]]; then
        echo ""
        echo "[SKIP] tide / $stem — not found in manifest: $MANIFEST"
        echo "       Run tide_sweep_train.sh first."
        FAIL=$((FAIL + 1))
        FAILED+=("tide / $stem (missing manifest entry)")
        continue
    fi

    checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"

    if [[ ! -d "$checkpoint" ]]; then
        echo ""
        echo "[SKIP] tide / $stem — checkpoint not found: $checkpoint"
        FAIL=$((FAIL + 1))
        FAILED+=("tide / $stem (missing checkpoint)")
        continue
    fi

    for dataset in "${datasets[@]}"; do
        label="tide / $stem / $dataset"
        echo ""
        echo "============================================================"
        echo "  Eval: $label"
        echo "  Checkpoint: $checkpoint"
        echo "============================================================"

        CMD=(
            "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
            --model tide
            --model-config "$model_config"
            --dataset "$dataset"
            --config-dir "$CONFIG_DIR"
            --checkpoint "$checkpoint"
            --context-length "$ctx_len"
            --forecast-length 96
            --cuda-device "$GPU"
            --probabilistic
        )
        if [[ -n "$cov_cols" ]]; then
            # word-split intentional: cov_cols is space-separated
            # shellcheck disable=SC2086
            CMD+=(--covariate-cols $cov_cols)
        fi

        if "${CMD[@]}"; then
            echo "[OK] $label"
            PASS=$((PASS + 1))
        else
            echo "[FAIL] $label"
            FAIL=$((FAIL + 1))
            FAILED+=("$label")
        fi
    done
done

echo ""
echo "============================================================"
echo "  TiDE sweep eval complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
