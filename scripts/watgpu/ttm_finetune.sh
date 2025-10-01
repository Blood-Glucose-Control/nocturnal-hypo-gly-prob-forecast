#!/bin/bash


# Navigate to ttm.py and configure the parameters for the run
# It can be finetuned or resume from a checkpoint


#SBATCH --job-name="gpu_test_ttm_finetune"
# Time format: HH:MM:SS
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=HI
#SBATCH --gres=gpu:1

#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

#SBATCH --mail-user=cjrisi@uwaterloo.ca,t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate

# Debug: Check environment
echo "=== Environment Check ==="
echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
which python
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"
## python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__} at {tf.__file__}')"
echo "=========================="

# Start the run
python $HOME/nocturnal/src/train/ttm.py


# Run sbatch ttm_finetune.sh to finetune the model in the terminal
