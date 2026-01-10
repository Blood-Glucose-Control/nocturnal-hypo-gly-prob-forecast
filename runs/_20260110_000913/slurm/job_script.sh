#!/bin/bash
#SBATCH --job-name=rl-insulin-_20260110_000913
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
#SBATCH --output=/u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/slurm/slurm-%j.out
#SBATCH --error=/u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/slurm/slurm-%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Run directory: /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913"
echo "Configuration: "

# Change to project directory
cd /u201/y329xie/nocturnal-hypo-gly-prob-forecast

# Activate virtual environment
echo "Activating virtual environment: env"
source env/bin/activate

# Save system information
echo "Python version: $(python --version)" > /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
echo "GPU information:" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
nvidia-smi >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt 2>/dev/null || echo "No GPU available" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
echo "Environment packages:" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
pip list >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt

# Save git information if available
if [[ -d .git ]]; then
    echo "Git commit: $(git rev-parse HEAD)" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
    echo "Git branch: $(git branch --show-current)" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
    echo "Git status:" >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
    git status --porcelain >> /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/logs/system_info.txt
fi


# Run the training
echo "Starting training..."
python /u201/y329xie/AdaPTS/scripts/run.py --forecast_horizon 24 --model_name "Autogluon/chronos-t5-small" --context_length 512 --seed 17966 --device "cuda:0" --dataset_name "Illness" --data-path /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/results/latest.csv --adapter "linearlVAE" --use_revin --supervised "True"

# # Optionally run prediction after training
# if [[ "false" == true ]]; then
#     echo "Running prediction after training..."
#     python src/main.py --cfg "" --mode predict
# fi

# # Save final model with run-specific name
# if [[ -f "ddpg_simglucose.zip" ]]; then
#     mv ddpg_simglucose.zip "/u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/checkpoints/final_model.zip"
# fi

# Move any other generated files to run directory
for file in *.pth *.txt _stable_baselines3_version; do
    if [[ -f "" ]]; then
        mv "" "/u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913/checkpoints/"
    fi
done

echo "Job completed at: $(date)"
echo "All outputs saved to: /u201/y329xie/nocturnal-hypo-gly-prob-forecast/runs/_20260110_000913"
