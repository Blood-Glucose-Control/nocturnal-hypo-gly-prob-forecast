#!/usr/bin/env bash
# Load readings one month at a time to avoid OOM. Each month runs in a separate
# psql process so memory is released between runs. Progress is logged with timestamps.
#
# Run from INSIDE the Apptainer container (after apptainer shell ...):
#   cd /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07
#   nohup ./insert_readings_month_by_month.sh > insert_readings_month_by_month.log 2>&1 &
#   tail -f insert_readings_month_by_month.log
#
# Or with custom log path:
#   ./insert_readings_month_by_month.sh /path/to/my.log
#
# Optional: sleep between months (default 60s) so DB can settle:
#   SLEEP=180 nohup ./insert_readings_month_by_month.sh > insert_readings_month_by_month.log 2>&1 &

# ps aux | grep psql
# tail -f insert_readings_month_by_month.log

set -euo pipefail

PSQL="${PSQL:-psql -h 127.0.0.1 -U postgres -d gluroo_datasets}"
BASE="/data/shared/cache/data/gluroo_2026/raw"
# skipping 2025-01 because it's already loaded
# MONTHS=(01 02 03 04 05 06 07 08 09 10 11 12)
MONTHS=(02 03 04 05 06 07 08 09 10 11 12)
TOTAL=${#MONTHS[@]}

log() { echo "[$(date -Iseconds)] $*"; }

n=0
for mm in "${MONTHS[@]}"; do
  n=$((n + 1))
  gz="${BASE}/readings-2025-${mm}-01.csv.gz"
  if [[ ! -f "$gz" ]]; then
    log "SKIP $gz (file not found)"
    continue
  fi
  log "Loading 2025-${mm} (${n}/${TOTAL})..."
  $PSQL -v ON_ERROR_STOP=1 -c "\COPY readings(gid,date,bgl,trend) FROM PROGRAM 'zcat ${gz}' WITH (FORMAT csv, HEADER true, DELIMITER ',');"
  log "Done 2025-${mm}."
  if [[ $n -lt $TOTAL ]]; then
    SLEEP="${SLEEP:-60}"
    log "Sleeping ${SLEEP}s before next month..."
    sleep "$SLEEP"
  fi
done

log "All ${TOTAL} months finished."
