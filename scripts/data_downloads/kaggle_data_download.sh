#!/bin/bash

# This script assumes the following relative paths:
# - The script is located at: scripts/data_downloads/kaggle_data_download.sh
# - The target directory for data is: src/data/kaggle_brisT1D
# - The relative path from the script to the target directory is: ../../src/data/kaggle_brisT1D

# Determine the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory relative to the script's location
TARGET_DIR="$SCRIPT_DIR/../../src/data/datasets/kaggle_bris_t1d/raw"

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Download the data from the Kaggle competition
kaggle competitions download -c brist1d -p $TARGET_DIR

# Unzip the downloaded data into the target directory
unzip "$TARGET_DIR/*.zip" -d $TARGET_DIR

# Remove the zip file after extraction
rm $TARGET_DIR/*.zip
