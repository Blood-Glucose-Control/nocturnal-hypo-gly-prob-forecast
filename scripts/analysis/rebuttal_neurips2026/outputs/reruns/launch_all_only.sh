#!/usr/bin/env bash
# Launch both GPU runners in detached tmux sessions (disconnect-safe).
cd "/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast"
tmux new-session -d -s rerun_only_gpu0 'bash scripts/analysis/rebuttal_neurips2026/outputs/reruns/run_gpu0_only.sh; exec bash'
tmux new-session -d -s rerun_only_gpu1 'bash scripts/analysis/rebuttal_neurips2026/outputs/reruns/run_gpu1_only.sh; exec bash'
tmux ls
