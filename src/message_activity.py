def get_min_messages(df, window_len = 6):
    '''
    Returns the minimum number of messages in a given window.

    Args:
        df (pd.DataFrame): The dataframe containing the messages.
        window_len (int): The length of the window to consider.

    Returns:
        min_messages (int): The minimum number of messages in the given window.
    '''
    min_messages = df["msg_type"].rolling(window=window_len).sum().min()
    return min_messages

