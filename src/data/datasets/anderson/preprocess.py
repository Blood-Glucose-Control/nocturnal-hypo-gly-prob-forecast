# This is the script for processing Anderson2016 data into the common format.
# Author: David Buchanan
# Adapted to Python by ChatGPT
# Original Date: January 31st, 2020, edited June 13th, 2020 by Elizabeth Chun

import os
import pandas as pd

# TODO: Set the dataset name (name of the created folder)
dataset = "../data_downloads/anderson"

# Change the working directory to the dataset folder
# os.chdir(dataset)

# Define the file path
file_path = os.path.join("../data_downloads/anderson", "Data Tables", "CGM.txt")

# Alternatively, if the file structure has been changed, place CGM.txt directly in the folder and use:
# file_path = "CGM.txt"

# Read the raw data
curr = pd.read_csv(file_path, sep="|")

# Reorder and keep only the columns we want
curr = curr.iloc[
    :, [0, 4, 3]
]  # Columns 1, 5, and 4 in R are 0, 4, and 3 in Python (0-indexed)

# Rename columns to standard format
curr.columns = ["id", "time", "gl"]

# Ensure glucose values are numeric
curr["gl"] = pd.to_numeric(curr["gl"], errors="coerce")

# Standardize date and time
curr["time"] = pd.to_datetime(curr["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# Define the output file name
# NOTE: saves csv file in this dir
output_file = f"./{dataset}_processed.csv"

# Check if the file exists to determine header inclusion
write_header = not os.path.exists(output_file)

# Save the cleaned data
curr.to_csv(output_file, index=False, header=write_header, mode="a")

# Note: 'DisplayTime' is used because it is user-configurable.
# "The time displayed to the user on the receiver or phone. This time is assumed to be user-configurable."
# Source: https://developer.dexcom.com/glossary
