# HOW TO RUN
# 1. Go to job.sh and double check the resources are set correctly
# 2. Add all the yaml files you want to run here
# 3. Give a description of the model run
    # The description should include:
    # 1. The main purpose of the run
    # 2. Why am I running this run
    # 3. What is the main change of this run
# 4. Run the script by typing "bash batch.sh"


# Change your yaml files here
yaml_files=(
    "0_naive_05min.yaml"
)

# Change your description here
msg="This is a test run"




# Get current timestamp in format YYYY-MM-DD_HH-MM-SS
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
init_msg="The run is initiated at $timestamp"
message="${init_msg}\n${msg}"


for yaml_file in "${yaml_files[@]}"; do
    sbatch job.sh "$yaml_file" "$message"
    echo "Submitted job for $yaml_file"
    sleep 1
done
