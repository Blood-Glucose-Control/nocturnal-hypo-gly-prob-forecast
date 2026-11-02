#!/usr/bin/env bash
# Publication-grade statistics run (patient-cluster is the primary/honest method).
# Launch in tmux: tmux new-session -d -s pubstats 'bash .../run_publication_stats.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /data/home/cjrisi/nocturnal)"
source .noctprob-venv/bin/activate 2>/dev/null
LOG=scripts/analysis/rebuttal_neurips2026/outputs/pubstats
mkdir -p "$LOG"
NB=10000

run () {  # name  module  extra-args...
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  python -m "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

echo "===== publication stats: n_boot=$NB ====="
# A7 first (deterministic scipy Friedman/Wilcoxon) + CD diagrams
run a7_rank      scripts.analysis.rebuttal_neurips2026.a7_rank_significance
run a7_cd        scripts.analysis.rebuttal_neurips2026.a7_cd_diagram
# Bootstrap analyses (patient-cluster is primary)
run a1_patient   scripts.analysis.rebuttal_neurips2026.a1_significance --bootstrap-unit patient --n-boot $NB
run a1_episode   scripts.analysis.rebuttal_neurips2026.a1_significance --bootstrap-unit episode --n-boot $NB
run a4_covariate scripts.analysis.rebuttal_neurips2026.a4_covariate --n-boot $NB
run a8_zeroshot  scripts.analysis.rebuttal_neurips2026.a8_zeroshot_significance --n-boot $NB
# A2 point + prob operating points with bootstrap CIs
run a2_point     scripts.analysis.rebuttal_neurips2026.a2_alarm --score point --bootstrap --n-boot 2000
echo "[$(date +%H:%M:%S)] ALL PUBLICATION STATS COMPLETE" | tee "$LOG/_ALL_DONE.txt"
