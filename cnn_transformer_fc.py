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
    """CNN operations for processing stacked radar chart features"""
    
    def __init__(self, input_channels: int = 45, feature_dim: int = 64):  # 15 timesteps * 3 RGB = 45 channels
        super().__init__()
        
        # CNN layers for processing stacked radar charts (all 15 timesteps as channels)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # Adaptive pooling to fixed size
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Feature mapping layer
        self.feature_map = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, channel_num, height, width]
               where channel_num = 15 timesteps * 3 RGB channels = 45
        Returns:
            features: Tensor of shape [batch_size, feature_dim]
        """
        # Apply CNN layers
        x = self.conv_layers(x)  # [batch_size, 256, 4, 4]
        
        # Flatten and apply feature mapping
        x = x.view(x.size(0), -1)  # [batch_size, 256*4*4]
        features = self.feature_map(x)  # [batch_size, feature_dim]
        
        return features


class SpatialTransformerEncoder(nn.Module):
    """Transformer encoder for spatial patches with positional encoding"""
    
    def __init__(self, 
                 input_dim: int = 64,  # CNN feature dimension
                 d_model: int = 96,  # Transformer dimension (divisible by 8)
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 384,
                 dropout: float = 0.1,
                 patch_size: int = 4):  # Size of spatial patches
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size  # 4x4 = 16 patches
        
        # Project input features to patches and then to transformer dimension
        self.patch_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for spatial patches
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim] - CNN features
        Returns:
            output: Tensor of shape [batch_size, d_model] - aggregated features
        """
        batch_size = x.size(0)
        
        # Create spatial patches by replicating the feature vector
        # This simulates spatial patches for transformer processing
        patches = x.unsqueeze(1).repeat(1, self.num_patches, 1)  # [batch_size, num_patches, input_dim]
        
        # Add some spatial variation to patches (optional enhancement)
        spatial_encoding = torch.randn(1, self.num_patches, x.size(1), device=x.device) * 0.1
        patches = patches + spatial_encoding
        
        # Project to transformer dimension
        patches = self.patch_projection(patches)  # [batch_size, num_patches, d_model]
        
        # Transpose for transformer (num_patches, batch_size, d_model)
        patches = patches.transpose(0, 1)
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Apply transformer encoder
        output = self.transformer_encoder(patches)  # [num_patches, batch_size, d_model]
        
        # Global average pooling across patches
        output = output.mean(dim=0)  # [batch_size, d_model]
        
        return output


class CNNTransformerFC(nn.Module):
    """Complete CNN + Transformer + FC model for precipitation prediction"""
    
    def __init__(self,
                 input_channels: int = 45,  # 15 timesteps * 3 RGB = 45 channels
                 cnn_feature_dim: int = 64,
                 transformer_dim: int = 96,
                 transformer_heads: int = 8,
                 transformer_layers: int = 6,
                 num_classes: int = 1,  # For regression, use 1; for classification, adjust accordingly
                 dropout: float = 0.1):
        super().__init__()
        
        # CNN feature extractor (processes all 15 timesteps as stacked channels)
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            feature_dim=cnn_feature_dim
        )
        
        # Spatial Transformer encoder
        self.transformer = SpatialTransformerEncoder(
            input_dim=cnn_feature_dim,
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, channel_num, height, width]
               where channel_num = 15 timesteps * 3 RGB channels = 45
               Represents 15 chronologically arranged radar charts stacked as channels
        Returns:
            output: Predictions of shape [batch_size, num_classes]
        """
        # Extract CNN features from stacked radar charts
        cnn_features = self.cnn_extractor(x)  # [batch_size, cnn_feature_dim]
        
        # Apply spatial transformer encoder
        transformer_output = self.transformer(cnn_features)  # [batch_size, transformer_dim]
        
        # Apply fully connected layers for final prediction
        output = self.fc_layers(transformer_output)  # [batch_size, num_classes]
        
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
        input_channels=config.get('input_channels', 45),  # 15 timesteps * 3 RGB
        cnn_feature_dim=config.get('cnn_feature_dim', 64),
        transformer_dim=config.get('transformer_dim', 96),
        transformer_heads=config.get('transformer_heads', 8),
        transformer_layers=config.get('transformer_layers', 6),
        num_classes=config.get('num_classes', 1),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Example usage and testing
    config = {
        'input_channels': 45,  # 15 timesteps * 3 RGB = 45 channels
        'cnn_feature_dim': 64,
        'transformer_dim': 96,  # Changed to 96 (divisible by 8)
        'transformer_heads': 8,
        'transformer_layers': 6,
        'num_classes': 1,
        'dropout': 0.1
    }
    
    model = create_model(config)
    
    # Test with dummy data - new format: [batch_size, channel_num, height, width]
    batch_size = 4
    channel_num = 45  # 15 timesteps * 3 RGB channels
    height = 64
    width = 64
    
    dummy_input = torch.randn(batch_size, channel_num, height, width)
    
    print("Model architecture:")
    print(model)
    print(f"\nInput shape: {dummy_input.shape}")
    print("Input format: [batch_size, channel_num, height, width]")
    print("where channel_num = 15 timesteps × 3 RGB channels = 45")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")