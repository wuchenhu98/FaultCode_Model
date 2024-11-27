import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from data_processing import load_and_process_data
import os

def save_predictions(file_path, model_path="models/fault_prediction_model.pkl", output_dir="results/predictions/"):
    """
    Load model, make predictions, and save results to a file in percentage format.

    Args:
        file_path (str): Path to the data CSV file.
        model_path (str): Path to the trained model file.
        output_dir (str): Directory to save the predictions file.

    Returns:
        None
    """
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load and preprocess the data
    hourly_features, hourly_targets = load_and_process_data(file_path)
    feature_columns = pd.read_csv(file_path).columns.tolist()[1:]  # Fault code names

    # Predict probabilities
    y_pred = model.predict(hourly_features)

    # Normalize predictions to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)

    # Reshape predictions for output (samples x 5 time windows x fault codes)
    y_pred_reshaped = y_pred.reshape(-1, 5, len(feature_columns))
    average_probs = y_pred_reshaped.mean(axis=0)  # Average across samples

    # Convert probabilities to percentage format with string formatting
    average_probs_percentage = pd.DataFrame(
        (average_probs * 100).round(4),
        columns=feature_columns
    ).applymap(lambda x: f"{x:.4f}%")  # Add percentage symbol and format

    # Prepare the output DataFrame
    time_windows = ["未来一小时", "未来两小时", "未来三小时", "未来四小时", "未来五小时"]
    output_df = average_probs_percentage
    output_df.insert(0, "未来时间窗口", time_windows)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the output file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = f"{output_dir}predictions_{timestamp}.csv"
    output_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {output_file_path}")

def save_correlation_matrix(file_path, output_dir="results/correlation/"):
    """
    Calculate and save the correlation matrix of fault codes as a CSV file.

    Args:
        file_path (str): Path to the data CSV file.
        output_dir (str): Directory to save the correlation matrix file.

    Returns:
        None
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Drop the timestamp column if present
    if 'ts' in data.columns:
        data = data.drop(columns=['ts'])

    # Calculate the correlation matrix and round to 4 decimal places
    correlation_matrix = data.corr().round(4)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the correlation matrix to a CSV file
    output_file_path = f"{output_dir}correlation_matrix.csv"
    correlation_matrix.to_csv(output_file_path, encoding='utf-8-sig')
    print(f"Correlation matrix saved to {output_file_path}")

# Example usage
if __name__ == "__main__":
    # Generate predictions
    save_predictions("data/device_fault_data.csv", "models/fault_prediction_model.pkl", "results/predictions/")

    # Generate fault code correlation matrix
    save_correlation_matrix("data/device_fault_data.csv", "results/correlation/")
