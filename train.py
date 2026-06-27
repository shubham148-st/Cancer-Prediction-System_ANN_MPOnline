"""
Model Training Script
Trains a simple Multilayer Perceptron (MLP) for binary disease classification using PyTorch.
Includes basic preprocessing, model definition, training loop, and evaluation.
"""

import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DiseasePredictionNet(nn.Module):
    """
    A simple 3-layer Feedforward Neural Network for binary classification.
    """
    def __init__(self, input_size):
        super(DiseasePredictionNet, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        # Second hidden layer
        self.fc2 = nn.Linear(32, 16)
        # Output layer
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_and_preprocess_data(dataset_path: str):
    """
    Loads data from CSV, splits it, and scales features for neural network ingestion.
    """
    logging.info(f"Loading data from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find {dataset_path}. Please run fetch_real_data.py first.")
        
    df = pd.read_csv(dataset_path)
    X = df.drop('Target', axis=1).values
    y = df['Target'].values

    logging.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural networks are sensitive to feature scales, so we standard scale them.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert everything to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

def train_model(args):
    """
    Main orchestrator function for training the model.
    """
    # 1. Prepare data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(args.data)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Model
    input_size = X_train.shape[1]
    model = DiseasePredictionNet(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. Training Loop
    logging.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch [{epoch+1}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # 4. Evaluation
    logging.info("Evaluating the model on the test set...")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = y_pred.round()
        correct = y_pred_class.eq(y_test).sum().item()
        total = y_test.shape[0]
        accuracy = correct / total
        logging.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{total} correct)")

    # 5. Artifact Saving
    logging.info("Saving model weights and preprocessing artifacts...")
    torch.save(model.state_dict(), args.model_out)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(input_size, 'input_size.pkl')
    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ANN for disease prediction.")
    parser.add_argument('--data', type=str, default='dataset.csv', help="Path to the training dataset.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument('--model_out', type=str, default='disease_model.pth', help="Path to save the trained model.")
    
    args = parser.parse_args()
    train_model(args)
