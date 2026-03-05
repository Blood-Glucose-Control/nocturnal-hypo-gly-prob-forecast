#!/bin/bash

# Local (non-SLURM) version of tamborlane data processing
# Run with: bash scripts/data_processing_scripts/tamborlane_data_processing_local.sh

set -e  # Exit on error

# Setup logging - create output directory and log files with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="cache/data/tamborlane_2008/data_processing_job_output"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/local-${TIMESTAMP}.out"
ERR_FILE="$OUTPUT_DIR/local-${TIMESTAMP}.err"

# Redirect stdout and stderr to files while also displaying to terminal
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERR_FILE" >&2)

echo "Log files: $LOG_FILE, $ERR_FILE"
echo "Started at: $(date)"

# Activate the virtual environment (adjust path as needed for your environment)
if [ -f "$HOME/nocturnal/.noctprob-venv/bin/activate" ]; then
    source $HOME/nocturnal/.noctprob-venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using currently active virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment found. Proceeding with system Python."
fi

# Inline Python code to process the tamborlane_2008 data
echo "Starting Tamborlane data processing"
python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='tamborlane_2008',
    use_cached=False,
    parallel=True,
    max_workers=14,
)"
echo "Tamborlane data processing completed"
echo "Finished at: $(date)"
