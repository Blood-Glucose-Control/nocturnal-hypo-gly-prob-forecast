# Data Download Scripts

## kaggle_brisT1D

Before following the steps below make sure your machine is configured to work with the kaggle public API:

[How to Use Kaggle - Public API](https://www.kaggle.com/docs/api)

To run the script from your main directory, use the following command:

```bash scripts/competition_submission/kaggle_submission.sh {SUBMISSON_CSV_DIRECTORY}```

where `SUBMISSION_CSV_DIRECTORY` is the directory of the file you want to submit.

Make sure you are in your main directory when you run this command. If the script is executable, you can also run it directly:

1. Make the script executable (if not already done):

```chmod +x scripts/data_downloads/kaggle_data_download.sh```

2. Run the script:

```./scripts/data_downloads/kaggle_data_download.sh```

You need to also make sure to have your local set-up with the kaggle API.
