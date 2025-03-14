#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=03:00:00
#SBATCH --mem=6GB
#SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=<your_email>@uwaterloo.ca
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Check if yaml file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: sbatch job.sh <yaml_config_file> <description> <timestamp>"
    echo "Example: sbatch job.sh 2_arch_EGARCH_05min.yaml 'This is a description of the model run'"
    exit 1
fi

source $HOME/nocturnal-hypo-gly-prob-forecast/.noctprob-venv/bin/activate

# Run the model with the provided yaml file
python $HOME/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/run_model.py "$1" "$2" "$3"
