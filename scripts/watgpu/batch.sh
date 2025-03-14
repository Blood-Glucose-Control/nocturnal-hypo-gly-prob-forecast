# Change your yaml files here
yaml_files=(
    "0_naive_05min.yaml"
    "0_naive_15min.yaml"
)

# Change your description of the run here
msg="This is a test run"


# Get current timestamp in ISO 8601 format
timestamp=$(date '+%Y-%m-%dT%H:%M:%S')
init_msg="The run is initiated at $timestamp"
message="${init_msg}\n${msg}"


for yaml_file in "${yaml_files[@]}"; do
    sbatch job.sh "$yaml_file" "$message" "$timestamp"
    echo "Submitted job for $yaml_file"
    sleep 1
done
