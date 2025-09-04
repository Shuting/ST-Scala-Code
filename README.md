# CNN+Transformer+FC Model

A PyTorch implementation of a CNN+Transformer+FC model for precipitation prediction based on radar chart sequences.

## Architecture

The model consists of three main components:

1. **CNN Feature Extractor**: Processes 15 chronologically arranged radar charts to extract spatial features
2. **Transformer Encoder**: Uses attention mechanisms to capture temporal dependencies with positional encoding
3. **Fully Connected Layer**: Maps the transformer output to final predictions

## Features

- Handles sequences of radar chart images (15 time steps)
- Position encoding for temporal awareness (EPOS units: 10 for embedding, 100 for transformer)
- Configurable transformer architecture (heads, layers, dimensions)
- Support for both regression and classification tasks
- Efficient CNN feature extraction with adaptive pooling

## Usage

### Basic Usage

```python
from cnn_transformer_fc import create_model

# Configure the model
config = {
    'input_channels': 3,        # RGB radar charts
    'sequence_length': 15,      # 15 time steps
    'cnn_feature_dim': 64,      # CNN output dimension
    'transformer_dim': 100,     # Transformer dimension
    'transformer_heads': 8,     # Number of attention heads
    'transformer_layers': 6,    # Number of transformer layers
    'num_classes': 1,          # Output classes (1 for regression)
    'dropout': 0.1             # Dropout rate
}

# Create model
model = create_model(config)

# Input: [batch_size, seq_len, channels, height, width]
# Output: [batch_size, num_classes]
```

### Training Example

```python
python example_usage.py
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Components

### CNNFeatureExtractor
- Processes individual radar charts
- Convolutional layers with ReLU activation
- Adaptive pooling for consistent feature sizes
- Feature mapping to desired dimensions

### TransformerEncoder
- Multi-head self-attention mechanism
- Positional encoding for sequence awareness
- Layer normalization and residual connections
- Configurable number of layers and attention heads

### Output Layer
- Fully connected layers with dropout
- Global average pooling across sequence dimension
- Configurable for regression or classification

## Architecture Details

Based on the paper architecture:
- Input: 15 chronologically arranged radar charts
- CNN operations for each dimension
- Position embedding (unit=10)
- Transformer with attention mechanisms (unit=100)
- Fully connected output layer

## Performance

The model is designed for:
- Precipitation prediction from radar sequences
- Efficient processing of temporal radar data
- Scalable to different sequence lengths and image sizes