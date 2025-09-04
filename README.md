# CNN+Transformer+FC Model

A PyTorch implementation of a CNN+Transformer+FC model for precipitation prediction based on radar chart sequences.

## Architecture

The model consists of three main components:

1. **CNN Feature Extractor**: Processes 15 chronologically arranged radar charts stacked as 45 input channels (15 × 3 RGB)
2. **Spatial Transformer Encoder**: Uses attention mechanisms to capture spatial feature relationships with positional encoding
3. **Fully Connected Layer**: Maps the transformer output to final predictions

## Features

- Handles radar chart data as stacked channels (45 channels = 15 time steps × 3 RGB)
- Spatial attention mechanism for feature enhancement
- Position encoding for spatial awareness
- Configurable transformer architecture (heads, layers, dimensions)
- Support for both regression and classification tasks
- Efficient CNN feature extraction with adaptive pooling

## Usage

### Basic Usage

```python
from cnn_transformer_fc import create_model

# Configure the model
config = {
    'input_channels': 45,       # 15 timesteps * 3 RGB = 45 channels
    'cnn_feature_dim': 64,      # CNN output dimension
    'transformer_dim': 96,      # Transformer dimension
    'transformer_heads': 8,     # Number of attention heads
    'transformer_layers': 6,    # Number of transformer layers
    'num_classes': 1,          # Output classes (1 for regression)
    'dropout': 0.1             # Dropout rate
}

# Create model
model = create_model(config)

# Input: [batch_size, channel_num, height, width]
# where channel_num = 15 timesteps × 3 RGB = 45
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
- Processes stacked radar charts (45 input channels)
- Convolutional layers with ReLU activation
- Adaptive pooling for consistent feature sizes
- Feature mapping to desired dimensions

### SpatialTransformerEncoder
- Multi-head self-attention mechanism for spatial features
- Positional encoding for spatial patch awareness
- Layer normalization and residual connections
- Configurable number of layers and attention heads

### Output Layer
- Fully connected layers with dropout
- Direct processing of transformer output
- Configurable for regression or classification

## Architecture Details

Based on the paper architecture:
- Input: 15 chronologically arranged radar charts stacked as 45 channels
- CNN operations processing all timesteps simultaneously
- Spatial transformer with positional encoding
- Transformer with attention mechanisms for spatial feature enhancement
- Fully connected output layer

## Performance

The model is designed for:
- Precipitation prediction from radar sequences
- Efficient processing of stacked temporal radar data
- Scalable to different channel configurations and image sizes