#!/usr/bin/env bash
# rerun_10_failed.sh
# Re-runs only the 10 jobs that failed in the previous rerun_best_models.sh run.
# Appends successes to rerun_manifest.txt; overwrites rerun_failed.txt with only
# still-failing jobs.
#
# Usage:
#   bash scripts/experiments/rerun_10_failed.sh
#   bash scripts/experiments/rerun_10_failed.sh --dry-run

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

DEFAULT_PYTHON="$REPO_DIR/.noctprob-venv/bin/python"
VENVS_DIR="$REPO_DIR/.venvs"
MANIFEST="$REPO_DIR/scripts/experiments/rerun_manifest.txt"
FAILED="$REPO_DIR/scripts/experiments/rerun_failed.txt"
LOG_DIR="$REPO_DIR/logs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY-RUN] Commands will be printed but not executed."
fi

mkdir -p "$LOG_DIR"
# Only clear the failed file; manifest gets appended
> "$FAILED"

# ---------------------------------------------------------------------------
# 10 failed jobs (corrected cov_bucket from prior run)
# ---------------------------------------------------------------------------
JOBS=(
  # chronos2 (needs iob + insulin_availability)
  "chronos2|chronos2|brown_2019|iob_ia|trained_models/artifacts/chronos2/2026-04-26_06:26_RID20260426_062650_516757_holdout_workflow/snapshots/step_20000/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-30_042011_brown_2019_finetuned"
  "chronos2|chronos2|lynch_2022|iob_ia|trained_models/artifacts/chronos2/2026-04-26_06:26_RID20260426_062650_516757_holdout_workflow/snapshots/step_20000/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-30_041451_lynch_2022_finetuned"
  # moirai (needs iob + insulin_availability)
  "moirai|moirai|aleppo_2017|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_05:44_RID20260417_054446_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_2045_aleppo_2017_finetuned"
  "moirai|moirai|brown_2019|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_05:44_RID20260417_054446_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_0653_brown_2019_finetuned"
  "moirai|moirai|lynch_2022|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_06:10_RID20260417_061030_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-23_0620_lynch_2022_finetuned"
  # statistical/AutoARIMA+IOB (does not support covariates — bg_only)
  "statistical|statistical/AutoARIMA+IOB|aleppo_2017|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044121_aleppo_2017_finetuned"
  "statistical|statistical/AutoARIMA+IOB|brown_2019|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044421_brown_2019_finetuned"
  "statistical|statistical/AutoARIMA+IOB|lynch_2022|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044634_lynch_2022_finetuned"
  # tide (does not support covariates — bg_only; config.json patched)
  "tide|tide|aleppo_2017|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_19:05_RID20260417_190554_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-17_2240_aleppo_2017_finetuned"
  "tide|tide|lynch_2022|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_19:50_RID20260417_195034_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-23_2016_lynch_2022_finetuned"
)

TOTAL=${#JOBS[@]}
echo "Total jobs to retry: $TOTAL"

AUTOGLUON_MODELS=("deepar" "patchtst" "tft" "naive_baseline" "statistical")

get_python() {
    local model="$1"
    local venv_python="$VENVS_DIR/$model/bin/python"
    if [[ -x "$venv_python" ]]; then
        echo "$venv_python"
        return
    fi
    for ag_model in "${AUTOGLUON_MODELS[@]}"; do
        if [[ "$model" == "$ag_model" ]]; then
            echo "$VENVS_DIR/chronos2/bin/python"
            return
        fi
    done
    echo "$DEFAULT_PYTHON"
}

build_cmd() {
    local entry="$1"
    local gpu="$2"
    IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
    local python_bin
    python_bin="$(get_python "$model_arg")"

    local ckpt_abs=""
    if [[ -n "$checkpoint" ]]; then
        if [[ "$checkpoint" = /* ]]; then
            ckpt_abs="$checkpoint"
        else
            ckpt_abs="$REPO_DIR/$checkpoint"
        fi
    fi

    local cov_args=""
    case "$cov_bucket" in
        iob_cob) cov_args="--covariate-cols iob cob" ;;
        iob_ia)  cov_args="--covariate-cols iob insulin_availability" ;;
        iob)     cov_args="--covariate-cols iob" ;;
        *)       cov_args="" ;;
    esac

    local prob_flag=""
    if [[ "$model_arg" != "moment" && "$model_arg" != "ttm" ]]; then
        prob_flag="--probabilistic"
    fi

    local ckpt_arg=""
    if [[ -n "$ckpt_abs" ]]; then
        ckpt_arg="--checkpoint $ckpt_abs"
    fi

    echo "$python_bin scripts/experiments/nocturnal_hypo_eval.py \
--model $model_arg \
--dataset $dataset \
--context-length 512 \
--forecast-length 96 \
--cuda-device $gpu \
--output-dir $run_path \
$ckpt_arg \
$cov_args \
$prob_flag"
}

# Preflight
echo ""
echo "=== Checkpoint preflight check ==="
preflight_ok=1
for entry in "${JOBS[@]}"; do
    IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
    if [[ -z "$checkpoint" ]]; then continue; fi
    if [[ "$checkpoint" = /* ]]; then ckpt_abs="$checkpoint"; else ckpt_abs="$REPO_DIR/$checkpoint"; fi
    if [[ ! -e "$ckpt_abs" ]]; then
        echo "  MISSING: $ckpt_abs  (for $model_csv / $dataset)"
        preflight_ok=0
    fi
done
if [[ $preflight_ok -eq 1 ]]; then echo "  All checkpoints found."; else echo ""; echo "ERROR: Fix missing checkpoints before running."; exit 1; fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "=== Dry-run: commands ==="
    idx=0
    for entry in "${JOBS[@]}"; do
        gpu=$(( (idx % 6) / 3 ))
        echo "--- Job $((idx+1))/$TOTAL (GPU $gpu) ---"
        build_cmd "$entry" "$gpu"
        echo ""
        (( idx++ )) || true
    done
    echo "[DRY-RUN] Done."
    exit 0
fi

echo ""
echo "=== Starting batch execution (6 jobs per batch) ==="

SUCCESS_COUNT=0
FAIL_COUNT=0

run_job() {
    local idx="$1"
    local entry="$2"
    local gpu="$3"
    local logfile="$LOG_DIR/rerun_retry_${idx}.log"
    local cmd
    cmd="$(build_cmd "$entry" "$gpu")"
    echo "[Job $((idx+1))/$TOTAL] GPU=$gpu → $logfile"
    eval "$cmd" > "$logfile" 2>&1
}

batch_start=0
while [[ $batch_start -lt $TOTAL ]]; do
    batch_end=$(( batch_start + 5 ))
    if [[ $batch_end -ge $TOTAL ]]; then batch_end=$(( TOTAL - 1 )); fi

    echo ""
    echo "--- Batch jobs $((batch_start+1))–$((batch_end+1)) ---"

    declare -A batch_pids
    declare -A batch_entries

    for (( i=batch_start; i<=batch_end; i++ )); do
        entry="${JOBS[$i]}"
        local_idx=$(( i - batch_start ))
        gpu=$(( local_idx / 3 ))
        run_job "$i" "$entry" "$gpu" &
        batch_pids[$i]=$!
        batch_entries[$i]="$entry"
    done

    for (( i=batch_start; i<=batch_end; i++ )); do
        pid=${batch_pids[$i]}
        entry="${batch_entries[$i]}"
        IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
        if wait "$pid"; then
            echo "  [OK]   Job $((i+1)): $model_csv / $dataset"
            echo "$run_path" >> "$MANIFEST"
            (( SUCCESS_COUNT++ )) || true
        else
            ec=$?
            echo "  [FAIL] Job $((i+1)): $model_csv / $dataset  (exit=$ec)"
            echo "${model_csv}|${dataset}|${checkpoint}|${run_path}|exit_code=${ec}" >> "$FAILED"
            (( FAIL_COUNT++ )) || true
        fi
    done

    batch_start=$(( batch_end + 1 ))
done

echo ""
echo "=== Done ==="
echo "  Succeeded: $SUCCESS_COUNT / $TOTAL"
echo "  Failed:    $FAIL_COUNT / $TOTAL"
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "  See $FAILED for details."
fi
