# loading signal traces for OSDM experiments
# Adam Lechowicz
# Sep 2025

import pandas as pd
import numpy as np

carbon_trace_paths = [
    "signal_traces/US-TEX-ERCO.csv",
    "signal_traces/US-CAL-CISO.csv"
]

price_trace_paths = [
    "signal_traces/CAISO-LMP-15min-2020-2025.csv",
    "signal_traces/PJM-LMP-15min-2020-2025.csv",
    "signal_traces/ERCOT-LMP-15min-2020-2025.csv",
    "signal_traces/ISONE-LMP-15min-2020-2025.csv"
]

def load_signal_trace(filename, month=1):
    """
    Load a signal trace from a CSV file.

    Args:
        filename (str): The name of the CSV file containing the signal trace.
    Returns:
        pd.Series: A pandas Series containing the signal trace data.
        p_min: float: The minimum value in the signal trace.
        p_max: float: The maximum value in the signal trace.
    """

    # find the right path for this filename
    path = "null"
    type = 0
    for p in carbon_trace_paths:
        if filename in p:
            path = p
            type = 1 #(carbon)
    for p in price_trace_paths:
        if filename in p:
            path = p
            type = 2 #(price)
    if path == "null":
        raise ValueError("Filename not found in predefined paths.")
    
    # load the CSV file into a DataFrame
    df = pd.read_csv(path)

    # parse the datetime column and set it as the index
    if type == 2:
        # rename the interval_start_utc column to datetime
        df.rename(columns={"interval_start_utc": "datetime"}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # extract the relevant signal column
    if type == 1:  # carbon
        signal = df["carbon_intensity_avg"]
    elif type == 2:  # price
        # filter the df to just consider the final year of data (2024)
        df = df[df.index.year == 2024]
        if month != 99:
            # filter the df to just consider the specified month
            df = df[df.index.month == month]
        # print the df.head
        # print(df.head())
        signal = df["lmp"]
        # if there are any negative prices, set them to one
        signal[signal < 1.0] = 1.0

    else:
        raise ValueError("Unknown signal type.")

    p_min = signal.min()
    # get the 99th percentile of the signal to avoid outliers
    p_99 = signal.quantile(0.99).copy()
    # cap the signal at the 99th percentile
    signal.loc[signal > p_99] = p_99
    p_max = signal.max()

    # extract the sequence of datetime indexes
    datetime_index = signal.index

    return signal, datetime_index, p_min, p_max
