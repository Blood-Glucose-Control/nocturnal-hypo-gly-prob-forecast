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

#SBATCH --mail-user=t3chan@uwaterloo.ca
#SBATCH --mail-type=ALL

# Activate the virtual environment
source \$HOME/nocturnal-hypo-gly-prob-forecast/.noctprob-venv/bin/activate

# Start the run
python \$HOME/nocturnal-hypo-gly-prob-forecast/src/train/ttm.py


# Run sbatch ttm_finetune.sh to finetune the model in the terminal
