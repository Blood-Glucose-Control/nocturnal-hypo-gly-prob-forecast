# Add all the yaml files you want to run here
# Go to job.sh and double check the resources are set correctly

yaml_files=(
    "2_arch_EGARCH_05min.yaml"
    "2_arch_GARCH_05min.yaml"
    "2_arch_HARCH_05min.yaml"
)

for yaml_file in "${yaml_files[@]}"; do
    sbatch job.sh "$yaml_file"
    echo "Submitted job for $yaml_file"
    sleep 1
done
