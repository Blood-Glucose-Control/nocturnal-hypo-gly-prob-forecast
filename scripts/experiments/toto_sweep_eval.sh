#!/usr/bin/env bash
# toto_sweep_eval.sh
#
# Runs nocturnal_hypo_eval.py --probabilistic for every Toto sweep config
# (00–08) × the appropriate dataset subset:
#   ALL (4 datasets): bg-only config (00)
#   IOB (3 datasets, excl. tamborlane): insulin covariate configs (01–04)
#   COB (2 datasets, excl. brown+tamborlane): carb covariate configs (05–08)
#
# Checkpoint paths are read from the manifest at:
#   trained_models/artifacts/toto/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, one entry per config)
# If a stem appears more than once (re-run), the last entry wins.
#
# If the manifest is missing this script auto-generates one by scanning
# trained_models/artifacts/toto/ for completed model.pt checkpoints and
# matching each run to a config stem via (covariate_cols, context_length, lr).
#
# Usage:
#   bash scripts/experiments/toto_sweep_eval.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/toto_sweep_eval.sh
#   bash scripts/experiments/toto_sweep_eval.sh 2>&1 | tee toto_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/toto/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
DATASETS_WITH_COB=(lynch_2022 aleppo_2017)
MANIFEST="trained_models/artifacts/toto/sweep_manifest.txt"

# Format: "stem|context_length|space-separated covariate cols (empty = BG only)|datasets_key"
CONFIG_META=(
    "00_bg_only|512||ALL"
    "01_bg_iob|512|iob|IOB"
    "02_bg_iob_ia|512|iob insulin_availability|IOB"
    "03_bg_iob_ia_high_lr|512|iob insulin_availability|IOB"
    "04_bg_iob_short_ctx|256|iob insulin_availability|IOB"
    "05_bg_iob_cob|512|iob cob|COB"
    "06_bg_full_features|512|iob cob insulin_availability carb_availability|COB"
    "07_bg_iob_cob_high_lr|512|iob cob|COB"
    "08_bg_iob_cob_short_ctx|256|iob cob|COB"
)

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: toto venv not found at $PYTHON"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-generate manifest from existing completed checkpoints (if missing).
# Matches each run dir to a config stem via (covariate_cols, context_length, lr).
# When multiple runs match the same stem, the most-recent timestamp wins.
# ---------------------------------------------------------------------------
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found — scanning trained_models/artifacts/toto/ for completed checkpoints..."
    ARTIFACTS_DIR="trained_models/artifacts/toto"
    "$PYTHON" - "$ARTIFACTS_DIR" "$MANIFEST" << 'PYEOF'
import sys, os, yaml
from pathlib import Path

artifacts_dir, manifest_path = sys.argv[1], sys.argv[2]

KEY_TO_STEM = {
    (frozenset(),                                                           512, round(1e-4, 7)): "00_bg_only",
    (frozenset(["iob"]),                                                    512, round(1e-4, 7)): "01_bg_iob",
    (frozenset(["iob", "insulin_availability"]),                            512, round(1e-4, 7)): "02_bg_iob_ia",
    (frozenset(["iob", "insulin_availability"]),                            512, round(5e-4, 7)): "03_bg_iob_ia_high_lr",
    (frozenset(["iob", "insulin_availability"]),                            256, round(1e-4, 7)): "04_bg_iob_short_ctx",
    (frozenset(["iob", "cob"]),                                             512, round(1e-4, 7)): "05_bg_iob_cob",
    (frozenset(["iob", "cob", "insulin_availability", "carb_availability"]),512, round(1e-4, 7)): "06_bg_full_features",
    (frozenset(["iob", "cob"]),                                             512, round(5e-4, 7)): "07_bg_iob_cob_high_lr",
    (frozenset(["iob", "cob"]),                                             256, round(1e-4, 7)): "08_bg_iob_cob_short_ctx",
}

# Collect best (most-recent) run per stem: stem -> (timestamp_str, out_dir_str)
best = {}
for run_dir in sorted(Path(artifacts_dir).iterdir()):
    if not run_dir.is_dir():
        continue
    if not (run_dir / "model.pt").is_dir():
        continue
    cfg_path = run_dir / "model_config.yaml"
    if not cfg_path.exists():
        continue
    cfg = yaml.safe_load(cfg_path.read_text())
    covs = frozenset(cfg.get("covariate_cols") or [])
    ctx = int(cfg.get("context_length", 512))
    lr = round(float(cfg.get("lr", 1e-4)), 7)
    key = (covs, ctx, lr)
    stem = KEY_TO_STEM.get(key)
    if stem is None:
        print(f"  WARN: cannot match {run_dir.name} (covs={set(covs)}, ctx={ctx}, lr={lr})",
              file=sys.stderr)
        continue
    # run_dir.name starts with timestamp (YYYY-MM-DD_HH:MM); sort lexicographically
    ts = run_dir.name[:16]
    if stem not in best or ts > best[stem][0]:
        best[stem] = (ts, str(run_dir))

if not best:
    print("ERROR: no completed toto checkpoints found", file=sys.stderr)
    sys.exit(1)

os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
with open(manifest_path, "w") as f:
    for stem, (ts, out_dir) in sorted(best.items()):
        f.write(f"{stem}\t{out_dir}\n")
        print(f"  {stem} -> {out_dir}")
print(f"Manifest written: {manifest_path} ({len(best)} entries)")
PYEOF
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run toto_sweep_train.sh first, or place completed runs under"
    echo "       trained_models/artifacts/toto/ and re-run this script."
    exit 1
fi

PASS=0
FAIL=0
FAILED=()

for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"
    model_config="configs/models/toto/${stem}.yaml"

    if [[ "$datasets_key" == "IOB" ]]; then
        datasets=("${DATASETS_WITH_IOB[@]}")
    elif [[ "$datasets_key" == "COB" ]]; then
        datasets=("${DATASETS_WITH_COB[@]}")
    else
        datasets=("${ALL_DATASETS[@]}")
    fi

    # Look up output dir from manifest (last match wins for re-runs)
    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"

    if [[ -z "$out_dir" ]]; then
        echo ""
        echo "[SKIP] toto / $stem — not found in manifest: $MANIFEST"
        echo "       Run toto_sweep_train.sh first."
        FAIL=$((FAIL + 1))
        FAILED+=("toto / $stem (missing manifest entry)")
        continue
    fi

    # Toto saves model.pt as a directory (base-class checkpoint format)
    checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"
    if [[ ! -d "$checkpoint" ]]; then
        echo ""
        echo "[SKIP] toto / $stem — checkpoint directory not found: $checkpoint"
        FAIL=$((FAIL + 1))
        FAILED+=("toto / $stem (missing checkpoint)")
        continue
    fi

    for dataset in "${datasets[@]}"; do
        label="toto / $stem / $dataset"
        echo ""
        echo "============================================================"
        echo "  Eval: $label"
        echo "  Checkpoint: $checkpoint"
        echo "============================================================"

        CMD=(
            "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
            --model toto
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
echo "  Toto sweep eval complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
