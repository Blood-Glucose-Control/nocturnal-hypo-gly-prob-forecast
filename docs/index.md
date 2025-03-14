# Welcome to Probabilistic Forecasting of Hypoglycemia Project
Documentation is currently under construction.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
    demo/ # The directory containing for producing the github pages demonstration of our work: https://blood-glucose-control.github.io/nocturnal-hypo-gly-prob-forecast/
    images/
    results/
    scripts/
        competition_submission/ # Scripts for submitting predictions to the BrisT1D Kaggle Competion.
        data_downloads/         # Scripts for downloading data from various sources to your local directory.
        models/                 #
        notebooks/
        watgpu/
    src/
    tests/
    .gitignore
    .pre-commit-config.yaml
    .readthedocs.yaml
    README.md
    requirements.txt
    setup.py

## MkDocs
This documentation is made with [mkdocs.org](https://www.mkdocs.org).

### Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.
