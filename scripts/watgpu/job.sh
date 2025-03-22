#!/bin/bash

# Specify your files and resources here
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
# Format: [yaml_file]="cores memory(GB) time"
# TODO: Add GPU resources once we need them
declare -A job_specs=(
    ["0_naive_05min.yaml"]="1 4 02:00:00"
    ["0_naive_15min.yaml"]="1 3 02:00:00"
)

# Set your email here for job notifications
email="t3chan@uwaterloo.ca"

# Change your description of the run here
description="This is a test run"



# Get current timestamp in ISO 8601 format
timestamp=$(date '+%Y-%m-%dT%H:%M:%S')
init_msg="The run is initiated at $timestamp"
message="${init_msg}\n${description}"

submit_job() {
    local yaml_file="$1"
    local description="$2"
    local timestamp="$3"
    local mem="$4"
    local cores="$5"
    local email="$6"
    local runtime="$7"

    # Create a temporary script with the proper SBATCH directives
    temp_script=$(mktemp -t temp_job.XXXXXX)
    cat << EOF > "${temp_script}"
#!/bin/bash
#SBATCH --time=$runtime
#SBATCH --cpus-per-task=$cores
#SBATCH --mem=${mem}GB
##SBATCH --gres=gpu:1

#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

#SBATCH --mail-user=$email
#SBATCH --mail-type=ALL

source \$HOME/nocturnal-hypo-gly-prob-forecast/.noctprob-venv/bin/activate

# Run the model with the provided yaml file
python \$HOME/nocturnal-hypo-gly-prob-forecast/scripts/watgpu/run_model.py "$yaml_file" "$description" "$timestamp"
EOF

    # Submit the temporary script to SLURM
    sbatch "${temp_script}"

    # Clean up the temporary script
    rm "${temp_script}"
}

# Process all YAML files
yaml_files=(${!job_specs[@]})
for yaml_file in "${yaml_files[@]}"; do
    read -r cores mem runtime <<< "${job_specs[$yaml_file]}"
    submit_job "$yaml_file" "$message" "$timestamp" "$mem" "$cores" "$email" "$runtime"
    echo "Submitted job for $yaml_file with $cores cores, ${mem}GB memory, and $runtime runtime"
    echo "----------------------------------------"
    sleep 1
done
