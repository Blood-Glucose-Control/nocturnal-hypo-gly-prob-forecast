# Welcome to Probabilistic Forecasting of Hypoglycemia Project

This project focuses on predicting the probability of nocturnal hypoglycemia events in individuals with diabetes. Using machine learning approaches, we aim to provide early warnings for these potentially dangerous low blood glucose events that occur during sleep.

Documentation is currently under construction.

## Project layout

    mkdocs.yml                  # The configuration file for this documentation.
    docs/                       # All subdirectories and files for the documentation site content.
        index.md                    # Makes this page
        requirements.txt            # mkdocs requirements file, not PFNH requirements.
        ...
    demo/                       # Directory containing for producing the github pages report of our work.
    images/                     # Images are used in READMEs, HowTos, Documentation, etc.
    models/                     # Storage for trained models.
        {datetime_model-desc}/      # Format for a trained model dir.
            configs/                # Config file to reproduce the trained model
            model/                  # Model serialization file storage. Loadable.
    results/
        figures/

    scripts/                    # Any sort of scripting file belongs here, usually for running experiments or notebooks.
        competition_submission/     # Scripts for submitting predictions to the BrisT1D Kaggle Competion.
        data_downloads/             # Scripts for downloading data from various sources to your local directory.
        notebooks/                  # .ipynb files, for naming convention see: https://cookiecutter-data-science.drivendata.org/using-the-template/
        watgpu/                     # bash or py scripts for training and evaluating models with WATGPU.
    src/
        data/                       # Data processing, downloading, and loading code
            carb_model/
            gluroo/
            insulin_model/
            kaggle_brisT1D/
            simglucose/
        eval/                       # Evaluation metrics and reporting
        tuning/                     # Hyperparameter tuning utilities
            configs/
        utils/                      # Utility functions and helper code
    tests/                      # Tests for functionality of .py files found in src/
    .gitignore                  # Files that we do not want tracked on github.
    .pre-commit-config.yaml     # Pre-commit should be run and pass all checks before commiting.
    .readthedocs.yaml           # Configuration file for this documentation.
    README.md                   # README.md for a repo's landing page.
    requirements.txt            # Requirements file to run all code on this repo.
    setup.py                    # For installing this repository locally.

## Installation

[Go to Installation Guide](user-guide/installation.md)

## Usage

[See Usage Instructions](user-guide/usage.md#basic-usage)

## MkDocs
This documentation is made with [mkdocs.org](https://www.mkdocs.org).

### Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.
