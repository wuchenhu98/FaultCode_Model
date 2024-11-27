import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from data_processing import load_and_process_data

# Define the LSTM model
class LSTMFaultPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMFaultPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Limit output to [0, 1]
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output
        out = out[:, -1, :]    # Get the last time step output
        out = self.fc(out)     # Fully connected layer
        return out

# Define custom dataset
class FaultDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def train_lstm_model(file_path, model_path="models/lstm_fault_model.pth", hidden_size=64, epochs=50, batch_size=32, learning_rate=0.001):
    # Load and preprocess data
    past_window = 1
    future_window = 5
    hourly_features, hourly_targets = load_and_process_data(file_path, past_window, future_window)

    # Normalize targets to [0, 1]
    hourly_targets = hourly_targets / hourly_targets.max()

    # Reshape inputs for LSTM: [samples, timesteps, features]
    X = hourly_features.reshape(hourly_features.shape[0], past_window, -1)
    y = hourly_targets.reshape(hourly_targets.shape[0], -1)

    # Split data into train and validation sets
    train_features, val_features, train_targets, val_targets = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader for PyTorch
    train_dataset = FaultDataset(train_features, train_targets)
    val_dataset = FaultDataset(val_features, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model, loss, and optimizer
    input_size = train_features.shape[2]
    output_size = train_targets.shape[1]
    model = LSTMFaultPredictor(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_lstm_model("data/device_fault_data.csv", "models/lstm_fault_model.pth")
