import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from data_processing import load_and_process_data

def train_and_save_model(file_path, model_path="models/fault_prediction_model.pkl"):
    """
    Train the model and save it to a file.

    Args:
        file_path (str): Path to the data CSV file.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load and preprocess data
    hourly_features, hourly_targets = load_and_process_data(file_path)
    hourly_targets = hourly_targets.reshape(hourly_targets.shape[0], -1)  # Flatten targets

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        hourly_features, hourly_targets, test_size=0.2, random_state=42
    )

    # Train model
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Training completed. RMSE: {rmse}")

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Example usage
if __name__ == "__main__":
    train_and_save_model("data/device_fault_data.csv", "models/fault_prediction_model.pkl")
