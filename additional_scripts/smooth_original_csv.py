import pandas as pd
import os
import sys

# only used for visualization in thesis

# Load the CSV file
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "original_csv"
chunks = os.listdir(path)
#name = 'button-press-topdown2'
for name in chunks:
    df = pd.read_csv(f'{path}/{name}') 

    # Calculate moving average
    window_size = 5  # Adjust the window size for smoothing
    df['Smoothed5Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 10  # Adjust the window size for smoothing
    df['Smoothed10Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 20  # Adjust the window size for smoothing
    df['Smoothed20Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 30  # Adjust the window size for smoothing
    df['Smoothed30Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 40  # Adjust the window size for smoothing
    df['Smoothed40Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 50  # Adjust the window size for smoothing
    df['Smoothed50Reward'] = df['Reward'].rolling(window=window_size, min_periods=1).mean()

    # Save the original and smoothed data to a new CSV
    df.to_csv(f'smoothed_csv/{name}', index=False)
