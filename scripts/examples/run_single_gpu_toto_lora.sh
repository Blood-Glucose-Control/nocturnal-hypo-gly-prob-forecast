#!/bin/bash
#SBATCH --job-name=toto_lora
#SBATCH --output=logs/toto_lora_%j.out
#SBATCH --error=logs/toto_lora_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# LoRA fine-tuning for Toto
# More parameter-efficient than full fine-tuning

echo "Starting Toto LoRA fine-tuning..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
date

# Activate environment (adjust as needed)
# source /path/to/your/env/bin/activate

# Run LoRA training
python scripts/examples/example_single_gpu_toto_lora.py

echo "LoRA training completed!"
date
