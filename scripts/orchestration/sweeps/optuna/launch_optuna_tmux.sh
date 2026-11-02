#!/usr/bin/env bash
# launch_optuna_tmux.sh
#
# Launch cooperative Chronos-2 x Optuna workers across GPUs in tmux.
#
# Each worker connects to the SAME shared SQLite study (config `storage`) and
# contributes trials. Workers are pinned to a GPU via CUDA_VISIBLE_DEVICES.
# TRIALS_PER_GPU workers are started per GPU (cap: 4). Chronos-2 LoRA at the
# challenge shape (context <= 288, 24-step forecast) is memory-light — measured
# peak VRAM on a 98 GB card:
#   batch 64 -> ~2.7 GiB   batch 128 -> ~3.8 GiB
#   batch 256 -> ~6.6 GiB  batch 512 -> ~15 GiB (compute-bound)
# Worst case (all workers at batch 512) with cap 4 is ~60 GiB, leaving headroom.
# Re-check with scripts/orchestration/sweeps/optuna/probe_vram.py if you widen
# the batch-size search.
#
# Usage:
#   GPUS="0 1" TRIALS_PER_GPU=1 N_TRIALS=40 \
#     bash scripts/orchestration/sweeps/optuna/launch_optuna_tmux.sh
#
#   # smoke run (tiny data + tiny steps, single worker):
#   SMOKE=1 GPUS="0" N_TRIALS=1 \
#     bash scripts/orchestration/sweeps/optuna/launch_optuna_tmux.sh
#
# Environment:
#   GPUS            Space-separated GPU ids            (default: auto-detect)
#   TRIALS_PER_GPU  Concurrent workers per GPU (1-4)   (default: 2)
#   N_TRIALS        Trials each worker runs            (default: 20)
#   CONFIG          Study YAML config                  (default: PoC config)
#   SESSION         tmux session name                  (default: optuna_chronos2)
#   PYTHON          Python interpreter                 (default: .venvs/autogluon)
#   NUM_THREADS     Per-worker CPU thread cap          (default: unset = no cap)
#                   Sets OMP/MKL/OpenBLAS/NumExpr/vecLib + torch intra-op threads.
#                   Prevents thread thrashing when running many workers: each
#                   AutoGluon/torch worker otherwise grabs ALL cores, so N
#                   workers oversubscribe CPU by ~N x. Benchmarked sweet spot for
#                   the IOBP2 shape is 2 (fits ~1.8x faster under 8-worker load).
#   SMOKE           1 = pass --smoke to workers        (default: 0)
#   TZ              Timezone for trial timestamps      (default: America/New_York)
#                   Optuna stores datetime.now() naively; this makes the
#                   dashboard "Started At" column read in your local zone.
#
# Monitor:   tmux attach -t optuna_chronos2
# Stop all:  tmux kill-session -t optuna_chronos2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

TRIALS_PER_GPU="${TRIALS_PER_GPU:-2}"
N_TRIALS="${N_TRIALS:-20}"
CONFIG="${CONFIG:-configs/experiments/nocturnal_forecast/chronos2_optuna_poc.yaml}"
SESSION="${SESSION:-optuna_chronos2}"
PYTHON="${PYTHON:-.venvs/autogluon/bin/python}"
# Sweep worker module: default is the Chronos-2 sweep; set to
# src.workflows.sweeps.optuna_tft.study for the TFT sweep.
MODULE="${MODULE:-src.workflows.sweeps.optuna_chronos2.study}"
SMOKE="${SMOKE:-0}"
# Per-worker CPU thread cap (unset = no cap). Exported into each worker command
# so OpenMP/MKL/BLAS/NumExpr and torch intra-op all honor it, preventing the
# N-workers-each-grab-all-cores thrash. torch reads OMP_NUM_THREADS at import.
NUM_THREADS="${NUM_THREADS:-}"
THREAD_ENV=""
if [[ -n "$NUM_THREADS" ]]; then
    THREAD_ENV="OMP_NUM_THREADS=${NUM_THREADS} MKL_NUM_THREADS=${NUM_THREADS} \
OPENBLAS_NUM_THREADS=${NUM_THREADS} NUMEXPR_NUM_THREADS=${NUM_THREADS} \
VECLIB_MAXIMUM_THREADS=${NUM_THREADS} "
fi
# Extra environment prefixed to each worker command (space-separated KEY=VAL
# pairs), e.g. EXTRA_ENV="TORCH_DISABLE_NATIVE_JIT=1" for the TimesFM 3.0 sweep.
EXTRA_ENV="${EXTRA_ENV:-}"
[[ -n "$EXTRA_ENV" ]] && EXTRA_ENV="${EXTRA_ENV} "
# Seconds between spawning workers. Larger values stagger heavy (GPU-bound
# fine-tune) workers so their startup CPU bursts (data/window building) do not
# collide. Default 2s suits light AutoGluon workers.
STAGGER_SECONDS="${STAGGER_SECONDS:-2}"
# Optuna stores trial start/complete times as naive datetime.now(); exporting a
# timezone makes the dashboard render them in local time (default: US Eastern).
TZ="${TZ:-America/New_York}"
export TZ

if [[ "$TRIALS_PER_GPU" -gt 4 ]]; then
    echo "[ERROR] TRIALS_PER_GPU=$TRIALS_PER_GPU exceeds the safety cap of 4." >&2
    echo "        Re-probe VRAM before raising this (batch 512 ~= 15 GiB/trial)." >&2
    exit 1
fi

# Detect GPUs if not provided.
if [[ -z "${GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
    else
        GPUS="0"
    fi
fi
GPUS="$(echo "$GPUS" | xargs)"  # trim

# ---------------------------------------------------------------------------
# Host-RAM guard. Each worker holds its data + AutoGluon training panel in
# SYSTEM RAM (not VRAM); over-subscribing RAM gets workers OOM-killed. Estimate
# demand and refuse to launch if it exceeds available RAM. Tune per config via
# MEM_PER_WORKER_GB (raise for larger patient_limit, lower for small subsets).
# ---------------------------------------------------------------------------
MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-35}"
N_GPUS="$(echo "$GPUS" | wc -w)"
N_WORKERS=$(( N_GPUS * TRIALS_PER_GPU ))
NEED_GB=$(( N_WORKERS * MEM_PER_WORKER_GB ))
AVAIL_GB="$(free -g | awk '/^Mem:/ {print $7}')"
if [[ -n "$AVAIL_GB" && "$NEED_GB" -gt "$AVAIL_GB" ]]; then
    echo "[ERROR] RAM guard: ${N_WORKERS} workers x ${MEM_PER_WORKER_GB}GB = ${NEED_GB}GB" >&2
    echo "        needed, but only ${AVAIL_GB}GB available. Workers would be OOM-killed." >&2
    echo "        Fix: lower TRIALS_PER_GPU, lower data.patient_limit in the config," >&2
    echo "        or (if you have measured headroom) lower MEM_PER_WORKER_GB." >&2
    exit 1
fi

SMOKE_FLAG=""
[[ "$SMOKE" == "1" ]] && SMOKE_FLAG="--smoke"

echo "================================================================"
echo "  Chronos-2 x Optuna tmux launcher"
echo "  GPUS            : $GPUS"
echo "  TRIALS_PER_GPU  : $TRIALS_PER_GPU"
echo "  WORKERS (total) : $N_WORKERS  (est ${NEED_GB}GB RAM / ${AVAIL_GB}GB avail)"
echo "  N_TRIALS/worker : $N_TRIALS"
echo "  NUM_THREADS     : ${NUM_THREADS:-<uncapped>}"
echo "  STAGGER_SECONDS : ${STAGGER_SECONDS}"
echo "  EXTRA_ENV       : ${EXTRA_ENV:-<none>}"
echo "  CONFIG          : $CONFIG"
echo "  SESSION         : $SESSION"
echo "  SMOKE           : $SMOKE"
echo "================================================================"

# Initialise the shared study once (idempotent).
echo "[1/2] Ensuring shared study exists..."
"$PYTHON" -m "$MODULE" create --config "$CONFIG" $SMOKE_FLAG

# Fresh tmux session (detached).
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" -n dashboard

echo "[2/2] Spawning workers..."
worker_idx=0
for gpu in $GPUS; do
    for slot in $(seq 1 "$TRIALS_PER_GPU"); do
        win="gpu${gpu}_w${slot}"
        cmd="TZ=${TZ} ${THREAD_ENV}${EXTRA_ENV}CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} -m ${MODULE} run \
--config ${CONFIG} --n-trials ${N_TRIALS} --gpu-label ${gpu} ${SMOKE_FLAG}"
        if [[ "$worker_idx" -eq 0 ]]; then
            tmux rename-window -t "${SESSION}:0" "$win"
            tmux send-keys -t "${SESSION}:${win}" "$cmd" C-m
        else
            tmux new-window -t "$SESSION" -n "$win"
            tmux send-keys -t "${SESSION}:${win}" "$cmd" C-m
        fi
        echo "  spawned worker on GPU ${gpu} (slot ${slot}) -> window ${win}"
        worker_idx=$((worker_idx + 1))
        sleep "$STAGGER_SECONDS"  # stagger worker starts (see STAGGER_SECONDS)
    done
done

echo ""
echo "Launched ${worker_idx} worker(s) in tmux session '${SESSION}'."
echo "  Attach : tmux attach -t ${SESSION}"
echo "  Stop   : tmux kill-session -t ${SESSION}"
