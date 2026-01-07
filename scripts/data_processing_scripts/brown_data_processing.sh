#!/bin/bash

#SBATCH --job-name="brown_data_processing"
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=HI
##SBATCH --gres=gpu:0
#SBATCH -o cache/data/brown_2019/data_processing_job_output/slurm-%j.out
#SBATCH -e cache/data/brown_2019/data_processing_job_output/slurm-%j.err
#SBATCH --mail-user=cjrisi@uwaterloo.ca,t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate


# Inline Python code to process the brown_2019 data (not the best practice but the task is simple enough)
echo "Starting Brown data processing"
python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='brown_2019',
    use_cached=False,
    parallel=True,
    max_workers=30,
)"
echo "Brown data processing completed"

## Run sbatch: sbatch scripts/data_processing_scripts/brown_data_processing.sh
