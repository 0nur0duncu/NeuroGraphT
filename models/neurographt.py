import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Literal
import math


class PositionalEncoding(nn.Module):
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


class CNNTransformerEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        transformer_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN Feature Extractor
        layers = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        # Transformer için projeksiyon
        self.input_projection = nn.Linear(conv_channels[-1], transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim, max_len=100, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = transformer_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # CNN feature extraction
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Projeksiyon ve pozisyonel kodlama
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        H = self.transformer(x)
        H = self.dropout(H)
        
        return H


class GraphBuilder(nn.Module):
    def __init__(
        self,
        num_nodes: int = 16,
        sparsity: float = 25.0,
        thresholding: str = "value"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.sparsity = sparsity
        self.thresholding = thresholding
    
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden = H.shape
        H_flat = H.reshape(batch_size, -1)
        total_features = H_flat.shape[1]
        
        features_per_node = total_features // self.num_nodes
        if features_per_node == 0:
            features_per_node = 1
            pad_size = self.num_nodes - total_features
            if pad_size > 0:
                H_flat = F.pad(H_flat, (0, pad_size))
        node_features = H_flat[:, :self.num_nodes * features_per_node]
        node_features = node_features.reshape(batch_size, self.num_nodes, -1)
        node_norm = node_features - node_features.mean(dim=-1, keepdim=True)
        node_std = node_norm.std(dim=-1, keepdim=True) + 1e-8
        node_norm = node_norm / node_std
        correlation = torch.bmm(node_norm, node_norm.transpose(-2, -1))
        correlation = correlation / node_features.shape[-1]
        if self.thresholding == "value":
            adjacency = self._value_threshold(correlation)
        else:
            adjacency = self._connection_threshold(correlation)
        
        return node_features, adjacency
    
    def _value_threshold(self, C: torch.Tensor) -> torch.Tensor:
        batch_size, N, _ = C.shape
        percentile = 100.0 - self.sparsity
        C_flat = C.reshape(batch_size, -1)
        k = max(1, int(percentile / 100.0 * C_flat.shape[1]))
        k = min(k, C_flat.shape[1] - 1)
        
        threshold, _ = torch.kthvalue(C_flat, k, dim=1)
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
        
        adjacency = (C > threshold).float()

        eye = torch.eye(N, device=C.device).unsqueeze(0)
        adjacency = adjacency * (1 - eye)
        
        return adjacency
    
    def _connection_threshold(self, C: torch.Tensor) -> torch.Tensor:
        batch_size, N, _ = C.shape
        n_connections = max(1, int(N * self.sparsity / 100.0))
        C_abs = torch.abs(C)
        eye = torch.eye(N, device=C.device).unsqueeze(0)
        C_abs = C_abs * (1 - eye)
        _, indices = torch.topk(C_abs, k=min(n_connections, N-1), dim=-1)
        
        adjacency = torch.zeros_like(C)
        adjacency.scatter_(2, indices, C.gather(2, indices))

        adjacency = torch.maximum(adjacency, adjacency.transpose(-2, -1))
        adjacency = adjacency * (1 - eye)
        
        return adjacency


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_channels
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            neighbor_sum = torch.bmm(adj, x)
            combined = x + neighbor_sum
            x = layer(combined)
            x = F.relu(x)
            x = self.dropout(x)
        x = x.max(dim=1)[0]
        
        return x


class NeuroGraphT(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        transformer_dim: int = 128,
        num_heads: int = 8,
        transformer_layers: int = 4,
        d_ff: int = 512,
        num_nodes: int = 16,
        sparsity: float = 25.0,
        thresholding: str = "value",
        gcn_hidden: int = 128,
        gcn_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cnn_transformer = CNNTransformerEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.graph_builder = GraphBuilder(
            num_nodes=num_nodes,
            sparsity=sparsity,
            thresholding=thresholding
        )
        
        # GCN input size hesapla
        seq_len = 64  # adaptive pool çıktısı
        total_features = seq_len * transformer_dim
        gcn_in_channels = total_features // num_nodes
        
        self.gcn = GCNEncoder(
            in_channels=gcn_in_channels,
            hidden_channels=gcn_hidden,
            num_layers=gcn_layers,
            dropout=dropout
        )
        
        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.dropout_rate = dropout

        self.classifier = nn.Linear(gcn_hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # CNN-Transformer encoding
        H = self.cnn_transformer(x)
        
        # Graph oluşturma
        node_features, adjacency = self.graph_builder(H)

        # GCN ile graf öğrenme
        graph_embedding = self.gcn(node_features, adjacency)

        # Sınıflandırma
        logits = self.classifier(graph_embedding)
        
        return logits
