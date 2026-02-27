#!/bin/bash
#SBATCH --job-name=tide_default
#SBATCH --output=logs/tide_default_%j.out
#SBATCH --error=logs/tide_default_%j.err
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# TiDE Validation - Default Configuration
# Tests TiDE's design (144 context, 4 dims, 1 layer)

echo "=========================================="
echo "TiDE Validation - Default Config"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd /u201/s6jindal/nocturnal-hypo-gly-prob-forecast || exit 1

# Activate conda environment
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

# Create logs directory
mkdir -p logs

# Run experiment
python scripts/tide_validation_experiment.py \
    --config default \
    --time-limit 14400 \
    --max-eval-episodes 500

echo ""
echo "End time: $(date)"
echo "=========================================="
