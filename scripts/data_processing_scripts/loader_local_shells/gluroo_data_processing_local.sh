#!/bin/bash

# Local (non-SLURM) version of gluroo data processing
# Go to project root
### WARNING: BEFORE STARTING, MAKE SURE THE PREVIOUS RUN ARE TERMINATED (ALL THE PROCESS ARE KILLED).
# To terminate the previous run, run:
# bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local_stop.sh (not tested yet)

## To start the processing using this script, run:
# conda activate noctprob
# nohup bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local.sh > gluroo_nohup_2026_03_17_0909.out 2>&1 &

# To stop the process, run:
# bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local_stop.sh (not tested yet)

# To view the process spawned by this script, run this in the terminal:
# ps -u t3chan -o pid,ppid,pcpu,pmem,stat,cmd --sort=ppid | grep python

set -e  # Exit on error

# Setup logging - create output directory and log files with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="cache/data/gluroo_2026/data_processing_job_output"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/local-${TIMESTAMP}.out"
ERR_FILE="$OUTPUT_DIR/local-${TIMESTAMP}.err"

# Redirect stdout and stderr to files while also displaying to terminal
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERR_FILE" >&2)

echo "Log files: $LOG_FILE, $ERR_FILE"
echo "Started at: $(date)"


# conda activate noctprob
PATIENTS_PER_BATCH=30
MAX_BATCHES_PER_RUN=10
export PATIENTS_PER_BATCH MAX_BATCHES_PER_RUN
echo "Starting Gluroo data processing (MAX_BATCHES_PER_RUN=$MAX_BATCHES_PER_RUN, patients_per_batch=$PATIENTS_PER_BATCH)"
python << 'PYEOF'
import os
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='gluroo_2026',
    keep_columns=None,
    use_cached=False,
    patients_per_batch=int(os.environ['PATIENTS_PER_BATCH']),
    patients_per_file=400, # Not important during processing, only when merging
    max_batches_per_run=int(os.environ['MAX_BATCHES_PER_RUN']),
    min_date_span_days=30,
    max_workers=20,
    load_all=False,
)
PYEOF
echo "Gluroo data processing completed"
echo "Finished at: $(date)"
