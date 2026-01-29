#!/bin/bash

#SBATCH --job-name="tamborlane_data_processing"
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20GB
#SBATCH --partition=HI
##SBATCH --gres=gpu:0
#SBATCH -o cache/data/tamborlane_2008/data_processing_job_output/slurm-%j.out
#SBATCH -e cache/data/tamborlane_2008/data_processing_job_output/slurm-%j.err
#SBATCH --mail-user=cjrisi@uwaterloo.ca,t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate


# Inline Python code to process the tamborlane_2008 data (not the best practice but the task is simple enough)
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

## Run sbatch: sbatch scripts/data_processing_scripts/tamborlane_data_processing.sh
