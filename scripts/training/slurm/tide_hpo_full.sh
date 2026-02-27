#!/bin/bash
#SBATCH --job-name=tide_hpo_full
#SBATCH --output=logs/tide_hpo_full_%j.out
#SBATCH --error=logs/tide_hpo_full_%j.err
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# TiDE HPO FULL SEARCH - WITH RAY BAYESIAN OPTIMIZATION
# 45 trials to match coverage: searches encoder/decoder dims [256,384,512] (~60% trials will fail)

echo "=========================================="
echo "TiDE HPO FULL SEARCH"
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
mkdir -p models/tide_hpo_full

# Run HPO experiment
echo "Starting TiDE FULL SEARCH HPO with Ray Bayesian optimization..."
echo ""

python scripts/tide_hpo_full_search.py \
    --num-trials 45 \
    --time-limit 18000 \
    --max-eval-episodes 500 \
    --output-dir models/tide_hpo_full

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Results saved to models/tide_hpo_full/"
else
    echo "FAILED: Check logs/tide_hpo_full_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
