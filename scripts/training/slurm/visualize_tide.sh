#!/bin/bash
#SBATCH --job-name=viz_tide
#SBATCH --output=logs/viz_tide_%j.out
#SBATCH --error=logs/viz_tide_%j.err
#SBATCH --partition=HI
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# TiDE Visualization - Best/Worst 30 Episodes

echo "=========================================="
echo "TiDE Visualization"
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

# Run visualization
python scripts/visualize_tide_best30.py

echo ""
echo "End time: $(date)"
echo "=========================================="
