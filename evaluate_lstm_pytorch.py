import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_lstm_pytorch import LSTMFaultPredictor, FaultDataset
from data_processing import load_and_process_data
from datetime import datetime

def save_lstm_predictions_and_correlation(file_path, model_path="models/lstm_fault_model.pth",
                                          prediction_dir="results/predictions/",
                                          correlation_dir="results/correlation/"):
    """
    Load LSTM model, make predictions, and save predictions & correlation matrix.

    Args:
        file_path (str): Path to the data CSV file.
        model_path (str): Path to the trained LSTM model file.
        prediction_dir (str): Directory to save the predictions file.
        correlation_dir (str): Directory to save the correlation matrix file.

    Returns:
        None
    """
    # Load and preprocess the data
    past_window = 1
    future_window = 5
    hourly_features, _ = load_and_process_data(file_path, past_window, future_window)

    # Reshape inputs for LSTM: [samples, timesteps, features]
    X = hourly_features.reshape(hourly_features.shape[0], past_window, -1)

    # Load the trained model
    input_size = X.shape[2]
    output_size = future_window * X.shape[2]
    hidden_size = 64  # Same as used during training
    model = LSTMFaultPredictor(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create DataLoader for prediction
    test_dataset = FaultDataset(X, np.zeros((X.shape[0], output_size)))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Predict
    all_predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            outputs = model(features)
            all_predictions.append(outputs.numpy())

    # Combine predictions
    all_predictions = np.vstack(all_predictions)

    # Clip predictions to ensure non-negative values
    all_predictions = np.clip(all_predictions, 0, None)

    # Reshape predictions for output
    y_pred_reshaped = all_predictions.reshape(-1, future_window, all_predictions.shape[1] // future_window)
    average_probs = y_pred_reshaped.mean(axis=0)

    # Prepare the predictions DataFrame
    feature_columns = pd.read_csv(file_path).columns.tolist()[1:]  # Fault code names
    time_windows = ["未来一小时", "未来两小时", "未来三小时", "未来四小时", "未来五小时"]
    predictions_df = pd.DataFrame(average_probs, columns=feature_columns)
    predictions_df.insert(0, "未来时间窗口", time_windows)

    # Save predictions
    os.makedirs(prediction_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = f"{prediction_dir}predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {predictions_file}")

    # Calculate and save correlation matrix
    correlation_matrix = predictions_df.iloc[:, 1:].corr().round(4)  # Exclude "未来时间窗口"
    os.makedirs(correlation_dir, exist_ok=True)
    correlation_file = f"{correlation_dir}correlation_matrix_{timestamp}.csv"
    correlation_matrix.to_csv(correlation_file, encoding='utf-8-sig')
    print(f"Correlation matrix saved to {correlation_file}")

if __name__ == "__main__":
    save_lstm_predictions_and_correlation("data/device_fault_data.csv", 
                                          "models/lstm_fault_model.pth", 
                                          "results/predictions/", 
                                          "results/correlation/")
