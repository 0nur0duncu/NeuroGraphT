import torch
import torch.nn as nn
import math
from typing import List


class PositionalEncoding(nn.Module):
    # Sinüzoidal Pozisyonel Kodlama.
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaselineCNNTransformer(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 5, 5],
        transformer_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN layers
        cnn_layers = []
        in_ch = in_channels
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        # Transformer için projeksiyon
        self.input_projection = nn.Linear(conv_channels[-1], transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim, max_len=100, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(transformer_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Projeksiyon ve pozisyonel kodlama
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Baseline1DCNNTransformer(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128, 64],
        kernel_sizes: List[int] = [7, 5, 5, 3],
        transformer_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN layers
        cnn_layers = []
        in_ch = in_channels
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        # Transformer için projeksiyon
        self.input_projection = nn.Linear(conv_channels[-1], transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim, max_len=100, dropout=dropout)
        
        # Transformer Encoder (daha derin)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(transformer_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Projeksiyon ve pozisyonel kodlama
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global max pooling
        x = x.max(dim=1)[0]
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x