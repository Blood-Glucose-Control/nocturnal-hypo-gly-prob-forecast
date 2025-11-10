import pandas as pd


def get_interval_minutes(processed_df: pd.DataFrame) -> int:
    """
    Get the interval of the time series data with datetime index.
    Args:
        processed_df: pd.DataFrame with datetime index
    Returns:
        int: Interval in minutes
    """
    if not isinstance(processed_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")
    if processed_df.shape[0] <= 1:
        raise ValueError("DataFrame must contain more than 1 row")
    return int((processed_df.index[1] - processed_df.index[0]).total_seconds() / 60)
