#!/bin/bash
# Convenience script for submitting ARIMA training jobs to SLURM

set -e  # Exit on any error

# Default values
CONFIG_FILE="scripts/training/configs/arima_config.yaml"
PATIENT_IDS=""
JOB_NAME="arima_train"
TEST_ONLY=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Submit ARIMA training job to SLURM cluster"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE      Configuration file (default: $CONFIG_FILE)"
    echo "  -p, --patients LIST    Space-separated patient IDs (default: all patients)"
    echo "  -n, --name NAME        Job name (default: $JOB_NAME)"
    echo "  -t, --test             Test submission without actually submitting"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Train on all patients with default config"
    echo "  $0 -c scripts/training/configs/autoarima_config.yaml"
    echo "  $0 -p \"patient_001 patient_002\"     # Train on specific patients"
    echo "  $0 -n my_arima_job                   # Custom job name"
    echo ""
    echo "Available config files:"
    ls scripts/training/configs/*arima*.yaml 2>/dev/null || echo "  No ARIMA configs found"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -p|--patients)
            PATIENT_IDS="$2"
            shift 2
            ;;
        -n|--name)
            JOB_NAME="$2"
            shift 2
            ;;
        -t|--test)
            TEST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate inputs
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available config files:"
    ls scripts/training/configs/*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "Submitting ARIMA training job to SLURM..."
echo "Config file: $CONFIG_FILE"
echo "Job name: $JOB_NAME"
if [ -n "$PATIENT_IDS" ]; then
    echo "Patients: $PATIENT_IDS"
else
    echo "Patients: all available"
fi

# Submit the job
if [ "$TEST_ONLY" = true ]; then
    echo "TEST MODE: Validating job submission (not actually submitting)..."
    if [ -n "$PATIENT_IDS" ]; then
        sbatch --test-only --job-name="$JOB_NAME" \
               --export=CONFIG_FILE="$CONFIG_FILE",PATIENT_IDS="$PATIENT_IDS" \
               scripts/slurm/train_cpu.sh
    else
        sbatch --test-only --job-name="$JOB_NAME" \
               --export=CONFIG_FILE="$CONFIG_FILE" \
               scripts/slurm/train_cpu.sh
    fi
    echo "Test completed successfully! The job would be accepted."
else
    if [ -n "$PATIENT_IDS" ]; then
        JOB_ID=$(sbatch --job-name="$JOB_NAME" \
                       --export=CONFIG_FILE="$CONFIG_FILE",PATIENT_IDS="$PATIENT_IDS" \
                       scripts/slurm/train_cpu.sh | awk '{print $4}')
    else
        JOB_ID=$(sbatch --job-name="$JOB_NAME" \
                       --export=CONFIG_FILE="$CONFIG_FILE" \
                       scripts/slurm/train_cpu.sh | awk '{print $4}')
    fi

    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor your job with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View logs with:"
    echo "  tail -f logs/${JOB_ID}_train_statistical.out"
    echo "  tail -f logs/${JOB_ID}_train_statistical.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
fi
