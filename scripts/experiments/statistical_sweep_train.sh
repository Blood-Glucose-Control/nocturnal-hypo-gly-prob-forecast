#!/usr/bin/env bash
# statistical_sweep_train.sh
#
# Trains all statistical baseline models (AutoARIMA, Theta, NPTS) on the
# appropriate dataset subsets. AutoGluon must fit these models to compute
# residual distributions for quantile synthesis.
#
# Configs:
#   00_autoarima_bg_only  — all 4 datasets
#   01_autoarima_bg_iob   — IOB datasets only (aleppo_2017, brown_2019, lynch_2022)
#   02_theta_bg_only      — all 4 datasets
#   03_npts_bg_only       — all 4 datasets
#
# Each config has a 2h time_limit. Series not fit in time fall back to Naive.
# CPU-only. No GPU required.
#
# Usage:
#   bash scripts/experiments/statistical_sweep_train.sh
#   bash scripts/experiments/statistical_sweep_train.sh 2>&1 | tee statistical_sweep_train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"
DATASETS_ALL="lynch_2022 aleppo_2017 brown_2019 tamborlane_2008"
DATASETS_WITH_IOB="lynch_2022 aleppo_2017 brown_2019"
MANIFEST_DIR="trained_models/artifacts/statistical"
MANIFEST="${MANIFEST_DIR}/sweep_manifest.txt"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
SKIP_STEPS="${SKIP_STEPS:-1 2 4 7}"
LOG_DIR="logs"

echo "=== Statistical sweep training  $(date) ==="
echo "  Venv: ${PYTHON}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

# Smoke-test imports
"$PYTHON" - <<'EOF'
from autogluon.timeseries.models import AutoARIMAModel, ThetaModel, NPTSModel
print("AutoGluon statistical model import OK (AutoARIMA, Theta, NPTS)")
EOF
echo ""

# Format: "config_path|datasets_key"
CONFIGS=(
    "configs/models/statistical/00_autoarima_bg_only.yaml|ALL"
    "configs/models/statistical/01_autoarima_bg_iob.yaml|IOB"
    "configs/models/statistical/02_theta_bg_only.yaml|ALL"
    "configs/models/statistical/03_npts_bg_only.yaml|ALL"
)

mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
touch "$MANIFEST"

PASS=0
FAIL=0
FAILED=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r config datasets_key <<< "$entry"
    stem="$(basename "$config" .yaml)"

    if [[ "$datasets_key" == "IOB" ]]; then
        datasets="$DATASETS_WITH_IOB"
    else
        datasets="$DATASETS_ALL"
    fi

    # Skip if already in manifest with a valid output dir
    existing_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST" 2>/dev/null || true)"
    if [[ -n "$existing_dir" && -d "${existing_dir}" ]]; then
        echo "[SKIP] ${stem} — already in manifest: $(basename "$existing_dir")"
        PASS=$(( PASS + 1 ))
        continue
    fi

    run_id="$(date +%Y%m%d_%H%M%S)_$$"
    out_dir="${MANIFEST_DIR}/$(date +%Y-%m-%d_%H%M)_RID${run_id}_${stem}"

    echo ""
    echo "============================================================"
    echo "  Training: statistical / ${stem}"
    echo "  Config:   ${config}"
    echo "  Datasets: ${datasets}"
    echo "  Output:   ${out_dir}"
    echo "  Note:     2h time_limit — unfit series fall back to Naive"
    echo "============================================================"

    if MODEL_TYPE="statistical" \
       VENV_NAME="chronos2" \
       MODEL_CONFIG="$config" \
       CONFIG_DIR="$CONFIG_DIR" \
       DATASETS="$datasets" \
       SKIP_TRAINING="false" \
       SKIP_STEPS="$SKIP_STEPS" \
       OUTPUT_BASE_DIR="$out_dir" \
       RUN_ID="$run_id" \
       "$WORKFLOW"; then
        echo "[OK] statistical / ${stem}"
        echo "${stem}"$'\t'"${out_dir}" >> "$MANIFEST"

        # Print leaderboard + per-series fallback count
        "$PYTHON" - "$out_dir" <<'PYEOF'
import sys, pathlib
try:
    from autogluon.timeseries import TimeSeriesPredictor
    predictor_dir = pathlib.Path(sys.argv[1])
    p = TimeSeriesPredictor.load(str(predictor_dir))
    print("\n--- AutoGluon leaderboard ---")
    print(p.leaderboard().to_string(index=False))
    info = p.info()
    print(f"\n--- Fit info ---")
    print(f"  num_train_rows : {info.get('num_train_rows', 'n/a')}")
    train_time = info.get('train_time')
    print(f"  train_time_s   : {train_time:.1f}" if isinstance(train_time, float) else f"  train_time_s   : {train_time}")

    # Per-series fallback count: AutoGluon local models (AutoARIMA, Theta, NPTS)
    # store one fitted object per item_id in model.local_models. Items that
    # timed out are replaced with a SimpleExpSmoothing/Naive fallback whose
    # class name differs from the primary model class.
    try:
        trainer = p._learner.load_trainer()
        for model_name in trainer.get_model_names():
            m = trainer.load_model(model_name)
            local = getattr(m, "local_models", None)
            if local:
                primary_cls = type(next(iter(local.values()))).__name__
                fallback_count = sum(
                    1 for v in local.values() if type(v).__name__ != primary_cls
                )
                total = len(local)
                print(f"\n  {model_name}: {total - fallback_count}/{total} series fit successfully "
                      f"({fallback_count} fell back to a simpler model)")
    except Exception as inner:
        print(f"  [fallback count unavailable: {inner}]")
        print("  (check the training log above for 'fallback' or 'time limit' messages)")

    print("-----------------------------\n")
except Exception as e:
    print(f"[WARN] Could not print leaderboard: {e}")
PYEOF

        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] statistical / ${stem}"
        FAIL=$(( FAIL + 1 ))
        FAILED+=("statistical / ${stem}")
    fi
done

echo ""
echo "============================================================"
echo "  Statistical sweep training complete  $(date)"
echo "  Passed: ${PASS} / $(( PASS + FAIL ))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
