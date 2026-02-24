#!/bin/bash
#SBATCH --job-name=tide_hpo
#SBATCH --output=logs/tide_hpo_%j.out
#SBATCH --error=logs/tide_hpo_%j.err
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# TiDE Hyperparameter Optimization - WITH RAY BAYESIAN OPTIMIZATION
# Tests 15 configurations with proper Bayesian search (encoder=decoder=256 fixed)

echo "=========================================="
echo "TiDE Hyperparameter Optimization"
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

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Create directories
mkdir -p logs
mkdir -p models/tide_hpo_ray

# Run HPO experiment
echo "Starting TiDE HPO experiment with Ray Bayesian optimization..."
echo ""

python scripts/tide_hpo_experiment.py \
    --num-trials 15 \
    --time-limit 18000 \
    --max-eval-episodes 500 \
    --output-dir models/tide_hpo_ray

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Results saved to models/tide_hpo_ray/"
else
    echo "FAILED: Check logs/tide_hpo_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
