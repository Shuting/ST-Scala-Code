"""
Example usage of the CNN+Transformer+FC model for precipitation prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cnn_transformer_fc import CNNTransformerFC, create_model


def create_dummy_dataset(num_samples: int = 1000, seq_len: int = 15, 
                        img_size: int = 64, channels: int = 3):
    """
    Create dummy radar chart dataset for demonstration
    
    Args:
        num_samples: Number of samples in dataset
        seq_len: Sequence length (number of time steps)
        img_size: Image size (height and width)
        channels: Number of input channels
    
    Returns:
        dataset: TensorDataset containing input sequences and targets
    """
    # Generate dummy radar chart sequences
    # In practice, this would be real radar data
    X = torch.randn(num_samples, seq_len, channels, img_size, img_size)
    
    # Generate dummy precipitation targets (regression task)
    # In practice, this would be real precipitation measurements
    y = torch.randn(num_samples, 1)
    
    return TensorDataset(X, y)


def train_model():
    """Example training loop"""
    
    # Model configuration
    config = {
        'input_channels': 3,
        'sequence_length': 15,
        'cnn_feature_dim': 64,
        'transformer_dim': 96,
        'transformer_heads': 8,
        'transformer_layers': 6,
        'num_classes': 1,  # Regression task
        'dropout': 0.1
    }
    
    # Create model
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = create_dummy_dataset(num_samples=800)
    val_dataset = create_dummy_dataset(num_samples=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  # For regression
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  New best model saved!')
        
        print('-' * 50)
    
    print(f'Training completed. Best validation loss: {best_val_loss:.6f}')


def inference_example():
    """Example inference with the trained model"""
    
    config = {
        'input_channels': 3,
        'sequence_length': 15,
        'cnn_feature_dim': 64,
        'transformer_dim': 96,
        'transformer_heads': 8,
        'transformer_layers': 6,
        'num_classes': 1,
        'dropout': 0.1
    }
    
    # Create model and load weights
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Loaded trained model weights")
    except FileNotFoundError:
        print("No trained model found, using random weights")
    
    model.eval()
    
    # Create sample input (15 radar charts)
    batch_size = 1
    seq_len = 15
    channels = 3
    img_size = 64
    
    sample_input = torch.randn(batch_size, seq_len, channels, img_size, img_size).to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(sample_input)
        
    print(f"Input shape: {sample_input.shape}")
    print(f"Prediction: {prediction.item():.6f}")
    
    return prediction


if __name__ == "__main__":
    print("CNN+Transformer+FC Model Example")
    print("=" * 50)
    
    # Run training example
    print("Starting training...")
    train_model()
    
    print("\nRunning inference example...")
    inference_example()