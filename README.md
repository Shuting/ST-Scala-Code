# CNN + Transformer + FC Model for Precipitation Prediction

This implementation provides a PyTorch model that combines Convolutional Neural Networks (CNN), Transformer encoders, and Fully Connected (FC) layers for precipitation prediction from radar data, based on the architecture described in Figure 6.

## Model Architecture

The model consists of three main components:

1. **CNN Feature Extractor**: Processes individual radar charts (9x9 grids) to extract spatial features
2. **Transformer Encoder**: Processes temporal sequences of features with attention mechanisms
3. **Fully Connected Layers**: Maps transformer outputs to final precipitation predictions

## Input/Output Specifications

- **Input Shape**: `(batch_size, channels, height, width)` = `(1024, 10, 9, 9)`
  - `batch_size`: Number of samples (1024)
  - `channels`: Number of radar channels/features (10)
  - `height, width`: Spatial dimensions of radar charts (9x9)

- **Output Shape**: `(batch_size,)` = `(1024,)`
  - Single real-valued precipitation prediction per sample

## Files

- `cnn_transformer_fc_model.py`: Main model implementation
- `train_example.py`: Training and evaluation example
- `requirements.txt`: Python dependencies

## Usage

### Basic Model Creation and Inference

```python
from cnn_transformer_fc_model import create_model
import torch

# Create model
model = create_model(input_channels=10, sequence_length=15)

# Create input tensor with specified shape
x = torch.randn(1024, 10, 9, 9)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(x)

print(f"Input shape: {x.shape}")   # torch.Size([1024, 10, 9, 9])
print(f"Output shape: {output.shape}")  # torch.Size([1024])
```

### Training Example

```python
from train_example import train_model, create_dummy_dataset
from cnn_transformer_fc_model import create_model

# Create model and data
model = create_model()
train_loader = create_dummy_dataset(num_samples=3000, batch_size=32)

# Train model
trained_model = train_model(model, train_loader, num_epochs=10)
```

## Model Components

### CNN Feature Extractor
- Processes 9x9 radar charts with 10 channels
- Uses 3 convolutional layers with ReLU activation
- Global average pooling for spatial dimension reduction
- Maps to 128-dimensional feature vectors

### Transformer Encoder
- Processes sequences of 15 time steps (chronologically arranged radar charts)
- Uses positional encoding for temporal information
- 6 transformer encoder layers with 8 attention heads
- 128-dimensional model with 512-dimensional feedforward networks

### Fully Connected Layers
- Two approaches implemented:
  1. Global pooling + FC (unit=100 as shown in figure)
  2. Sequence flattening + multiple FC layers
- Outputs single real value for precipitation prediction

## Model Parameters

- Total parameters: ~2.4M
- All parameters are trainable
- Dropout rate: 0.1 for regularization

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Code

```bash
# Test model implementation
python cnn_transformer_fc_model.py

# Run training example
python train_example.py
```

## Architecture Details

The model follows the structure shown in Figure 6:

1. **Input Layer**: 15 chronologically arranged radar charts (9x9 each)
2. **CNN Operations**: Feature extraction for each radar chart
3. **Feature Input/Map**: Mapping to consistent feature dimensions
4. **Position Embedding**: Temporal position encoding (unit = 10)
5. **EPOS Blocks**: Transformer encoder layers with attention
6. **Mix Information**: Feature mixing and attention mechanisms
7. **FC Layer**: Final fully connected layer (unit = 100)
8. **Output**: Single real-valued prediction

## Customization

The model can be customized by modifying parameters in the `create_model()` function:

- `input_channels`: Number of radar channels
- `sequence_length`: Number of time steps
- `cnn_output_features`: CNN feature dimension
- `transformer_d_model`: Transformer model dimension
- `transformer_nhead`: Number of attention heads
- `transformer_layers`: Number of transformer layers

## Notes

- The model handles both single-frame and sequence inputs
- Position encoding is applied for temporal sequence processing
- The implementation includes both training and inference modes
- GPU acceleration is supported when available