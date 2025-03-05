from src.data.data_loader import BrisT1DDataLoader
from src.tuning.benchmark import impute_missing_values
from sktime.forecasting.arima import AutoARIMA
from sktime.utils import mlflow_sktime

PATIENT = "p04"
MODEL = "AutoARIMA"

# Split up data into day and night
loader = BrisT1DDataLoader(use_cached=True)

train_df = loader.train_data[loader.train_data["p_num"] == PATIENT]
test_df = loader.validation_data[loader.validation_data["p_num"] == PATIENT]

TIME_STEP_SIZE = (
    train_df["datetime"].iloc[1] - train_df["datetime"].iloc[0]
).components.minutes

if TIME_STEP_SIZE != 5 and TIME_STEP_SIZE != 15:
    error = """
    First time step is not 5 or 15 minutes. Look at the most common time step size.
    """


def reduce_features(df):
    # Make sure index is set to datetime
    p_df = df.iloc[:]

    # Reduce features
    y_feature = ["bg-0:00"]
    x_features = [
        "hr-0:00",  # -> has too many NaNs
        "steps-0:00",
        "cals-0:00",
        "cob",
        "carb_availability",
        "insulin_availability",
        "iob",
    ]
    p_df = p_df[x_features + y_feature]

    # Impute with default methods
    p_df = impute_missing_values(p_df, columns=x_features)
    p_df = impute_missing_values(p_df, columns=y_feature)

    y, X = p_df[y_feature], p_df[x_features]
    return y, X


y_train, X_train = reduce_features(train_df)

forecaster = AutoARIMA(
    start_p=2,
    max_p=216,
    start_q=2,
    max_q=216,
    seasonal=False,
    n_jobs=-1,
    suppress_warnings=True,
)

forecaster.fit(y=y_train, X=X_train)

print(forecaster.get_params())
print(forecaster.get_fitted_params())

mlflow_sktime.save_model(sktime_model=forecaster, path="models/AutoARIMA")

forecaster = mlflow_sktime.load_model(model_uri="models/AutoARIMA")
