!!! warning "Under Construction"
    This documentation is currently under active development and subject to change.
    Some sections may be incomplete or missing.

## Training Models <a id="model-training"></a>
For most of our modeling tasks, the training falls into two broad categories, the models training goal, either single patient or multi-patient, and the model type, which can be either statistical, ml, dl, or foundation based models.
### Training Goal
#### Single Patient Training
Train one model for each patient.
The goal is to create a model that minimizes it's error as much as possible for a single patient.
Often it makes sense to take an existing multi-patient model and fine-tune it for a specific patient.

#### Multi-Patient Training
Train one model for all patients.
The goal is to create a generic model that works as well as possible for all patients.

### Model Type
#### Statistical Modeling
Currently, all of our *statistical* modeling is done with ```sktime```.
Essentially, these mostly fall under classic statistical time series techniques, almost anything that is not deep learning or a foundation model falls under this category, e.g.,: naive, ARIMA, exoponential, etc
We want to exhaustively find the best modeling parameters for each family of models.

#### Machine Learning
Models that aren't statistical modelling or deep learning, for e.g.,: Gradient Boosting, Tree-based algorithms, etc.

#### Deep Learning
Currently, all of our *deep learning* modeling is done with ```sktime``` wrappers, but this will be expanding to other time series libraries, usually within the ```sktime``` ecosystem like ```pytorch forecasting```.
Excluding foundation models, all neural network model training belongs here.

#### Foundational
There are two main foundation model use cases with our projects:
1. Fine-tuning existing foundation models with diabetes data
    - We plan to try LagLLama, Moirai, Chronos, TimesFM, TimeGPT, TimeGPT-Long, TinyTimeMixer, and potentially others.
2. Training dibaetes foundation models from scratch with as much diabetes data that we can find.
