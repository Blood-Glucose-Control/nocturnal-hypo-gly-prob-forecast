#!/bin/bash

set -e

# check for usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <submission-file>"
    exit 1
fi


COMPETITION_NAME="brist1d"
SUBMISSION_FILE="$1"
SUBMISSION_MESSAGE="Nocturnal BGL team!"

# check if kaggle CLI available
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Install it with 'pip install kaggle' and configure it."
    exit 1
fi

# check if valid submission file
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "Error: Submission file '$SUBMISSION_FILE' not found!"
    exit 1
fi

# submit to Kaggle
echo "Submitting '$SUBMISSION_FILE' to Kaggle competition '$COMPETITION_NAME'..."
kaggle competitions submit -c brist1d -f "$SUBMISSION_FILE" -m "$SUBMISSION_MESSAGE"
echo "Submission completed!"
