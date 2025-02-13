#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=00:00:10
#SBATCH --mem=3GB
#SBATCH --cpus-per-task=4
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
source $HOME/bgc/nocturnal-hypo-gly-prob-forecast/.noctprob-venv/bin/activate

# Task to run

# RUN THE PYTHON SCRIPT OVER HERE
python $HOME/bgc/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/cores.py --device gpu
python $HOME/bgc/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/cores.py --device cpu


# ENTER `sbatch cores.sh` to run the script
