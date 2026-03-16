#!/bin/bash

# Local (non-SLURM) version of gluroo data processing
# Go to project root
# conda activate noctprob
# bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local.sh
# or run in background with nohup
# nohup bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local.sh > gluroo_nohup_2026_02_24_0827.out 2>&1 &

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

PATIENTS_PER_BATCH=100
NUMBER_OF_PATIENTS=$((PATIENTS_PER_BATCH * 10))
export PATIENTS_PER_BATCH NUMBER_OF_PATIENTS
echo "Starting Gluroo data processing (NUMBER_OF_PATIENTS=$NUMBER_OF_PATIENTS)"
python << 'PYEOF'
import os
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='gluroo_2026',
    keep_columns=None,
    use_cached=False,
    patients_per_batch=int(os.environ['PATIENTS_PER_BATCH']),
    patients_per_file=400,
    number_of_patients_to_process=int(os.environ['NUMBER_OF_PATIENTS']),
    min_date_span_days=30,
    max_workers=10,
    load_all=True,
)
PYEOF
echo "Gluroo data processing completed"
echo "Finished at: $(date)"
