# Welcome to Probabilistic Forecasting of Hypoglycemia Project
Documentation is currently under construction.

## Project layout

    mkdocs.yml                  # The configuration file.
    docs/
        user-guide/
            installation.md
            usage.md
            watgpu-benchmark.md
        index.md                # The documentation homepage.
        api-reference.md        # Other markdown pages, images and other files.
        contributing.md
        requirements.txt        # mkdocs requirements file, not PFNH requirements.
    demo/                       # The directory containing for producing the github pages demonstration of our work.
    images/                     # Images are used in READMEs, HowTos, Documenteation, etc.
    models/                     # Storage for trained models
        {datetime_model-desc}/      # format for a trained model dir
            configs/                # the config file to reproduce the trained model
            model/                  # the model serialization file storage. Loadable.
    results/

    scripts/
        competition_submission/ # Scripts for submitting predictions to the BrisT1D Kaggle Competion.
        data_downloads/         # Scripts for downloading data from various sources to your local directory.
        models/                 #
        notebooks/
        watgpu/
    src/
    tests/                      # tests for functionality of .py files found in src/
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
