#!/usr/bin/env bash
# naive_baseline_sweep_train.sh
#
# Trains (fits) Naive and Average baseline models on all 4 datasets.
# AutoGluon fits Naive/Average in seconds; the main cost is data loading.
#
# Configs:
#   00_naive.yaml    — BG-only (Naive last-value), all 4 datasets
#   01_average.yaml  — BG-only (Average mean), all 4 datasets
#
# Both configs use no covariates. tamborlane_2008 is included (BG-only is fine).
#
# After training, a manifest is written so naive_baseline_sweep_eval.sh can
# locate the checkpoints for re-evaluation:
#   trained_models/artifacts/naive_baseline/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, last entry per stem wins)
#
# CPU-only. No GPU required.
#
# Usage:
#   bash scripts/experiments/naive_baseline_sweep_train.sh
#   bash scripts/experiments/naive_baseline_sweep_train.sh 2>&1 | tee logs/naive_train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ALL_DATASETS="lynch_2022 aleppo_2017 brown_2019 tamborlane_2008"
CONFIG_DIR="configs/data/holdout_10pct"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
MANIFEST="trained_models/artifacts/naive_baseline/sweep_manifest.txt"

# Configs: "config_path" (all use ALL datasets, BG-only)
CONFIGS=(
    "configs/models/naive_baseline/00_naive.yaml"
    "configs/models/naive_baseline/01_average.yaml"
)

mkdir -p "trained_models/artifacts/naive_baseline"
mkdir -p "logs"
touch "$MANIFEST"

PASS=0
FAIL=0
FAILED=()

for config in "${CONFIGS[@]}"; do
    stem="$(basename "$config" .yaml)"

    # Generate run ID matching workflow convention: YYYYMMDD_HHMMSS_PID
    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
    out_dir="trained_models/artifacts/naive_baseline/$(date +%Y-%m-%d_%H:%M)_RID${RUN_ID}_holdout_workflow"

    echo ""
    echo "============================================================"
    echo "  Training: naive_baseline / $stem"
    echo "  Config:   $config"
    echo "  Datasets: $ALL_DATASETS"
    echo "  Output:   $out_dir"
    echo "============================================================"

    if MODEL_TYPE="naive_baseline" \
       VENV_NAME="chronos2" \
       MODEL_CONFIG="$config" \
       CONFIG_DIR="$CONFIG_DIR" \
       DATASETS="$ALL_DATASETS" \
       SKIP_TRAINING="false" \
       SKIP_STEPS="1 2 4 7" \
       OUTPUT_BASE_DIR="$out_dir" \
       RUN_ID="$RUN_ID" \
       "$WORKFLOW"; then
        echo "[OK] naive_baseline / $stem"
        echo "${stem}	${out_dir}" >> "$MANIFEST"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] naive_baseline / $stem"
        FAIL=$((FAIL + 1))
        FAILED+=("naive_baseline / $stem")
    fi
done

echo ""
echo "============================================================"
echo "  Naive baseline training complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "  Manifest: $MANIFEST"
echo "============================================================"

[[ $FAIL -eq 0 ]]
