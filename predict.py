"""
Prediction Script
Loads a trained PyTorch model and executes inference on sample data.
"""

import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class DiseasePredictionNet(nn.Module):
    """
    Network definition must match the exact architecture used during training.
    """
    def __init__(self, input_size):
        super(DiseasePredictionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def run_inference(dataset_path: str, model_path: str):
    """
    Loads necessary artifacts and predicts classes for a few sample rows.
    """
    # 1. Validate artifacts exist
    for path in [dataset_path, model_path, 'scaler.pkl', 'input_size.pkl']:
        if not os.path.exists(path):
            logging.error(f"Missing required file: {path}")
            return

    # 2. Load artifacts
    scaler = joblib.load('scaler.pkl')
    input_size = joblib.load('input_size.pkl')
    
    model = DiseasePredictionNet(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. Load sample data
    # For demonstration, we'll grab the first two rows from the existing dataset
    df = pd.read_csv(dataset_path)
    X_sample = df.drop('Target', axis=1).iloc[:2].values
    
    # 4. Preprocess and Predict
    X_scaled = scaler.transform(X_sample)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # 5. Output Results
    print("-" * 50)
    print("Inference Results:")
    print("Class mapping: 1 = Benign, 0 = Malignant")
    print("-" * 50)
    
    for i, prob in enumerate(predictions):
        probability = prob.item()
        pred_class = "Benign" if probability >= 0.5 else "Malignant"
        actual_class = "Benign" if df['Target'].iloc[i] == 1 else "Malignant"
        
        logging.info(f"Sample {i+1}:")
        logging.info(f"  -> Predicted Probability (Benign): {probability:.4f}")
        logging.info(f"  -> Predicted Class: {pred_class}")
        logging.info(f"  -> Actual Class:    {actual_class}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the trained disease prediction model.")
    parser.add_argument('--data', type=str, default='dataset.csv', help="Path to CSV file containing sample data.")
    parser.add_argument('--model', type=str, default='disease_model.pth', help="Path to the saved PyTorch model.")
    
    args = parser.parse_args()
    run_inference(args.data, args.model)
