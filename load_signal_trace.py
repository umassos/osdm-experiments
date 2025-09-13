# loading signal traces for OSDM experiments
# Adam Lechowicz
# Sep 2025

import pandas as pd
import numpy as np

carbon_trace_paths = [
    "signal_traces/CAISO_5min_2022.csv",
    "signal_traces/CAISO_5min_2023.csv",
    "signal_traces/CAISO_5min_2024.csv"
]

price_trace_paths = [
    "signal_traces/CAISO_LMP_2022.csv",
    "signal_traces/CAISO_LMP_2023.csv",
    "signal_traces/CAISO_LMP_2024.csv"
]

def load_signal_trace(filename):
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

    # extract the relevant signal column
    if type == 1:  # carbon
        signal = df["carbon_signal"]
    elif type == 2:  # price
        signal = df["price_signal"]
    else:
        raise ValueError("Unknown signal type.")

    p_min = signal.min()
    p_max = signal.max()
    return signal, p_min, p_max
