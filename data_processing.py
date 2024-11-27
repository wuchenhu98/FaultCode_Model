import pandas as pd
import numpy as np

def load_and_process_data(file_path, past_window=1, future_window=5):
    """
    Load and preprocess data from the given file path.
    Args:
        file_path (str): Path to the CSV file.
        past_window (int): Number of past hours to use as input.
        future_window (int): Number of future hours to predict.

    Returns:
        hourly_features (np.ndarray): Features for training/testing.
        hourly_targets (np.ndarray): Targets for training/testing.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Check for missing values in the data
    if data.isnull().values.any():
        print("Warning: Data contains NaN values. Filling with 0.")
        data.fillna(0, inplace=True)

    # Convert timestamp and resample to hourly frequency
    data['ts'] = pd.to_datetime(data['ts'], format='%Y/%m/%d %H:%M')
    data.set_index('ts', inplace=True)
    hourly_data = data.resample('H').sum()

    # Normalize the data (convert counts to probabilities in [0, 1])
    hourly_data = hourly_data / hourly_data.max()
    hourly_data.fillna(0, inplace=True)  # Ensure no NaN after normalization

    # Generate sliding windows
    def generate_hourly_windows(df):
        features, targets = [], []
        for i in range(len(df) - past_window - future_window + 1):
            feature_window = df.iloc[i:i + past_window].values.flatten()
            target_window = df.iloc[i + past_window:i + past_window + future_window].values
            features.append(feature_window)
            targets.append(target_window)
        return np.array(features), np.array(targets)

    hourly_features, hourly_targets = generate_hourly_windows(hourly_data)
    return hourly_features, hourly_targets
