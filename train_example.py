"""
Training example for CNN+Transformer+FC model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cnn_transformer_fc_model import CNNTransformerFC, create_model


def create_dummy_dataset(num_samples=5000, batch_size=32):
    """Create dummy dataset for training/testing"""
    # Generate random radar chart data
    X = torch.randn(num_samples, 10, 9, 9)  # (samples, channels, height, width)
    
    # Generate dummy target values (e.g., precipitation predictions)
    # Using a simple function of the input for demonstration
    y = torch.randn(num_samples) * 0.5 + torch.mean(X.view(num_samples, -1), dim=1) * 0.1
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')
        scheduler.step()
    
    print("Training completed!")
    return model


def evaluate_model(model, test_loader):
    """Evaluate the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
    
    avg_loss = total_loss / num_samples
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss


if __name__ == "__main__":
    print("Creating CNN+Transformer+FC model for training...")
    
    # Create model
    model = create_model(input_channels=10, sequence_length=15)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_loader = create_dummy_dataset(num_samples=3000, batch_size=32)
    test_loader = create_dummy_dataset(num_samples=1000, batch_size=32)
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, num_epochs=5, learning_rate=0.001)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(trained_model, test_loader)
    
    # Save model
    print("\nSaving model...")
    torch.save(trained_model.state_dict(), '/workspace/cnn_transformer_fc_model.pth')
    print("Model saved to cnn_transformer_fc_model.pth")
    
    # Test inference on specified input shape
    print("\nTesting inference on specified input shape (1024, 10, 9, 9)...")
    test_input = torch.randn(1024, 10, 9, 9)
    trained_model.eval()
    with torch.no_grad():
        output = trained_model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output values: {output[:10]}")
    print(f"Output statistics - Mean: {output.mean():.4f}, Std: {output.std():.4f}")