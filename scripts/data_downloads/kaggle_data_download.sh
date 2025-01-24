#!/bin/bash

# Define the target directory
TARGET_DIR="src/data/kaggle_brisT1D"

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Download the data from the Kaggle competition
kaggle competitions download -c brist1d -p $TARGET_DIR

# Unzip the downloaded data into the target directory
unzip "$TARGET_DIR/*.zip" -d $TARGET_DIR

# Remove the zip file after extraction
rm $TARGET_DIR/*.zip
