#!/bin/bash

#SBATCH --job-name="aleppo_data_processing"
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=HI
##SBATCH --gres=gpu:1
#SBATCH -o results/runs/aleppo_data_processing/slurm-%j.out
#SBATCH -e results/runs/aleppo_data_processing/slurm-%j.err
#SBATCH --mail-user=cjrisi@uwaterloo.ca,t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate


# Inline Python code to process the aleppo data (not the best practice but the task is simple enough)
echo "Starting aleppo data processing"
python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='aleppo',
    train_percentage=0.9,
    use_cached=False,
    parallel=True,
    max_workers=10,
)
"
echo "Aleppo data processing completed"
