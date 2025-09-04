import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class CNNFeatureExtractor(nn.Module):
    """CNN operations for processing radar chart features"""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        
        # CNN layers for processing radar charts
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # Adaptive pooling to fixed size
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Feature mapping layer
        self.feature_map = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
        Returns:
            features: Tensor of shape [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all images in batch
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Flatten and apply feature mapping
        x = x.view(batch_size * seq_len, -1)
        features = self.feature_map(x)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        return features


class TransformerEncoder(nn.Module):
    """Transformer encoder with positional encoding"""
    
    def __init__(self, 
                 d_model: int = 96,  # Changed to 96 (divisible by 8)
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 384,  # Adjusted proportionally
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model)
        
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
        
        # Layer to project CNN features to transformer dimension
        self.input_projection = nn.Linear(64, d_model)  # From CNN feature_dim to d_model
        
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            src_key_padding_mask: Mask for padding tokens
        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Project to transformer dimension
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Transpose back to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)
        
        return output


class CNNTransformerFC(nn.Module):
    """Complete CNN + Transformer + FC model for precipitation prediction"""
    
    def __init__(self,
                 input_channels: int = 3,
                 sequence_length: int = 15,
                 cnn_feature_dim: int = 64,
                 transformer_dim: int = 96,
                 transformer_heads: int = 8,
                 transformer_layers: int = 6,
                 num_classes: int = 1,  # For regression, use 1; for classification, adjust accordingly
                 dropout: float = 0.1):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            feature_dim=cnn_feature_dim
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Fully connected output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Global pooling strategy for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
               Represents 15 chronologically arranged radar charts
            mask: Optional mask for padding sequences
        Returns:
            output: Predictions of shape [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Extract CNN features from radar charts
        cnn_features = self.cnn_extractor(x)  # [batch_size, seq_len, cnn_feature_dim]
        
        # Apply transformer encoder
        transformer_output = self.transformer(cnn_features, src_key_padding_mask=mask)
        # [batch_size, seq_len, transformer_dim]
        
        # Global pooling across sequence dimension
        # Transpose for pooling: [batch_size, transformer_dim, seq_len]
        pooled = transformer_output.transpose(1, 2)
        pooled = self.global_pool(pooled)  # [batch_size, transformer_dim, 1]
        pooled = pooled.squeeze(-1)  # [batch_size, transformer_dim]
        
        # Apply fully connected layers for final prediction
        output = self.fc_layers(pooled)  # [batch_size, num_classes]
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention weights from a specific transformer layer
        
        Args:
            x: Input tensor
            layer_idx: Which transformer layer to extract weights from (-1 for last layer)
        Returns:
            attention_weights: Attention weights tensor
        """
        # This would require modifying the transformer to return attention weights
        # For now, this is a placeholder for future implementation
        pass


def create_model(config: dict) -> CNNTransformerFC:
    """
    Factory function to create the model with configuration
    
    Args:
        config: Dictionary containing model configuration
    Returns:
        model: Initialized CNNTransformerFC model
    """
    return CNNTransformerFC(
        input_channels=config.get('input_channels', 3),
        sequence_length=config.get('sequence_length', 15),
        cnn_feature_dim=config.get('cnn_feature_dim', 64),
        transformer_dim=config.get('transformer_dim', 100),
        transformer_heads=config.get('transformer_heads', 8),
        transformer_layers=config.get('transformer_layers', 6),
        num_classes=config.get('num_classes', 1),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Example usage and testing
    config = {
        'input_channels': 3,
        'sequence_length': 15,
        'cnn_feature_dim': 64,
        'transformer_dim': 96,  # Changed to 96 (divisible by 8)
        'transformer_heads': 8,
        'transformer_layers': 6,
        'num_classes': 1,
        'dropout': 0.1
    }
    
    model = create_model(config)
    
    # Test with dummy data
    batch_size = 4
    seq_len = 15
    channels = 3
    height = 64
    width = 64
    
    dummy_input = torch.randn(batch_size, seq_len, channels, height, width)
    
    print("Model architecture:")
    print(model)
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")