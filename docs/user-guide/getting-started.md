!!! warning "Under Construction"
    This documentation is currently under active development and subject to change.
    Some sections may be incomplete or missing.

There are a variety of use cases for this repository.

The primary objective:

    - long-range (6-8 hours) probabilistic forecasting of hypoglycemia.

The secondary objectives:

    - short range forecasting projects
    - Benchmarking
      - 1 hour forecast Kaggle submission automation of BrisT1D contest
      - 1 hour forecast benchmark
      - 2 hour forecast benchmark
      - 6 hour probabilistic forecasting benchmark
      - 8 hour probabilistic forecasting benchmark
    - Training and evaluating diabetes foundation models for forecasting

## Setting up your project

### Local repo
This is just creates a local version of this repository so you can start making your own changes
1. In your local terminal, navigate to your desired project directory.
2. Run the following command: `git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast.git`

### Local environment
This ensures that everyone is working with the same tools and that the code you write will run on everyone else's machine, including our model training servers.

3. Create a virtual environment with python 3.11 in your terminal in the projects root directory

    `python3.11 -m venv .noctprob-venv`

    - **NOTE:** If you're using a different .venv name, please ensure it's added to the .gitignore file and not pushed to our repo.
4. Activate the venv:
    **Ubuntu:** `source .noctprob-venv/bin/activate`
    **Windows:** `.noctprob-venv\Scripts\Activate.ps1`
  - Your terminal should now show the venv being active:
    - `(.noctprob-venv) cjrisi@scslt520:~/Projects/diabetes/nocturnal-hypo-gly-prob-forecast$ `
5. Now update your venv with the required packages for our project:

    `pip install -r requirements.txt`
    It's good to run this command after you pull from the remote repo and there were changes because we may have added new packages; it takes the longest to run the first time, but it's usually quick.

#### pip-install package
We've made this repo pip-installable we highly recommend you install and never use local file paths. Everything you need should be importable from /src/.

6. In the project directory run `pip install -e .`
  - Read more about this [here](https://goodresearch.dev/setup#create-a-pip-installable-package-recommended).

Congratulations! You're ready to start contributing to our project.

---

## Pull Requests
Please run pre-commit and resolve any errors before making a PR.
### Run pre-commit
`pre-commit run --all-files`

### Testing
`pytest -v`


## Training Models
Please see our [model training section](modeling/model-training.md#model-training) for advanced usage.

A basic example is provided in this notebook:
- [Add link to this later]()

## Running Benchmarks
Please see our [model benchmarking section](benchmarking/model-benchmarking.md#model-benchmarking) for advanced usage.

A basic example of running our benchmarks is provided in this notebook:
- [Add link to this later]()
