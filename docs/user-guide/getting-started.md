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

## Training Models
### Regular Modeling
Currently, all of our *regular* modeling is done with ```sktime```.
Essentially anything that is not deep learning or a foundation model falls under this category.
We want to exhaustively find the best modeling parameters for each family of models.
#### Single Patient Training
Train one model for each patient.

#### Multiple Patient Training
Train one model for all patients.

### Deep Learning Modeling
Currently, all of our *deep learning* modeling is done with ```sktime``` wrappers, but this will be expanding to other time series libraries, usually within the ```sktime``` ecosystem like ```pytorch forecasting```.
Excluding foundation models, all neural network model training belongs here.

### Foundation Modeling
There are two main foundation model use cases with our projects:
1. Fine-tuning existing foundation models with diabetes data
2. Training dibaetes foundation models from scratch with as much diabetes data that we can find.

## Running Benchmarks
