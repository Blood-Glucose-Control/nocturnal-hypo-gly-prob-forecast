#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=01:00:00
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=t3chan@uwaterloo.ca
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
source $HOME/nocturnal-hypo-gly-prob-forecast/.noctprob-venv/bin/activate

# Task to run
# 5-min interval
# python $HOME/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/run_arima_5.py

# 15-min interval
python $HOME/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/run_arima_15.py
