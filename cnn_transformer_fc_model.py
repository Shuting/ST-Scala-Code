import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class CNNFeatureExtractor(nn.Module):
    """CNN operations for each radar chart"""
    
    def __init__(self, input_channels=10, output_features=128):
        super(CNNFeatureExtractor, self).__init__()
        
        # CNN layers for feature extraction from 9x9 radar charts
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        
        # Feature mapping to desired output dimension
        self.feature_map = nn.Linear(128, output_features)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        Returns:
            features: Tensor of shape (batch_size, output_features)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        # Feature mapping
        features = self.feature_map(x)
        
        return features


class TransformerEncoder(nn.Module):
    """Transformer encoder block (EPOS in the figure)"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, src):
        """
        Args:
            src: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            output: Tensor of shape (seq_len, batch_size, d_model)
        """
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        
        return output


class CNNTransformerFC(nn.Module):
    """Complete CNN + Transformer + FC model"""
    
    def __init__(self, 
                 input_channels=10, 
                 sequence_length=15,  # 15 chronologically arranged radar charts
                 cnn_output_features=128,
                 transformer_d_model=128,
                 transformer_nhead=8,
                 transformer_layers=6,
                 fc_hidden_dim=512,
                 dropout=0.1):
        super(CNNTransformerFC, self).__init__()
        
        self.sequence_length = sequence_length
        self.cnn_output_features = cnn_output_features
        
        # CNN feature extractor for each radar chart
        self.cnn_feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            output_features=cnn_output_features
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(transformer_d_model * sequence_length, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1)  # Output single real value
        )
        
        # Alternative: Global pooling + FC (as shown in figure with unit=100)
        self.global_pool_fc = nn.Sequential(
            nn.Linear(transformer_d_model, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )
        
        self.use_global_pooling = True  # Switch between two FC approaches
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, channels, height, width)
               For input shape (1024, 10, 9, 9), we need to reshape to handle sequence
        Returns:
            output: Tensor of shape (batch_size, 1) - single real value per sample
        """
        batch_size = x.size(0)
        
        # If input is (batch_size, channels, height, width), we need to create sequence dimension
        if len(x.shape) == 4:
            # Assume we're processing a single time step, replicate for sequence
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1, 1, 1)
        
        # Extract features from each radar chart in the sequence
        sequence_features = []
        for t in range(self.sequence_length):
            # Extract features for time step t
            chart_features = self.cnn_feature_extractor(x[:, t])  # (batch_size, cnn_output_features)
            sequence_features.append(chart_features)
        
        # Stack sequence features: (sequence_length, batch_size, features)
        sequence_features = torch.stack(sequence_features, dim=0)
        
        # Pass through transformer
        transformer_output = self.transformer(sequence_features)
        # transformer_output: (sequence_length, batch_size, d_model)
        
        if self.use_global_pooling:
            # Global average pooling over sequence dimension
            pooled_features = torch.mean(transformer_output, dim=0)  # (batch_size, d_model)
            output = self.global_pool_fc(pooled_features)
        else:
            # Flatten sequence and features for FC layers
            flattened = transformer_output.transpose(0, 1).contiguous()  # (batch_size, seq_len, d_model)
            flattened = flattened.view(batch_size, -1)  # (batch_size, seq_len * d_model)
            output = self.fc_layers(flattened)
        
        return output.squeeze(-1)  # Remove last dimension to return (batch_size,)


def create_model(input_channels=10, sequence_length=15):
    """Factory function to create the model with default parameters"""
    return CNNTransformerFC(
        input_channels=input_channels,
        sequence_length=sequence_length,
        cnn_output_features=128,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_layers=6,
        fc_hidden_dim=512,
        dropout=0.1
    )


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_model(input_channels=10, sequence_length=15)
    
    # Print model summary
    print("CNN + Transformer + FC Model")
    print("="*50)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # Test with the specified input shape: (1024, 10, 9, 9)
    batch_size = 1024
    channels = 10
    height = 9
    width = 9
    
    # Create dummy input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample values: {output[:5]}")
    
    # Test with sequence input as well
    sequence_length = 15
    x_sequence = torch.randn(batch_size, sequence_length, channels, height, width)
    print(f"\nSequence input shape: {x_sequence.shape}")
    
    with torch.no_grad():
        output_sequence = model(x_sequence)
    
    print(f"Sequence output shape: {output_sequence.shape}")
    print(f"Sequence output sample values: {output_sequence[:5]}")