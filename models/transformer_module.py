import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pozisyonel kodlama matrisi oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model num_heads'e tam bölünmeli"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Yorumlanabilirlik için
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V hesapla
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # Yorumlanabilirlik için sakla
        attention = self.dropout(attention)
        
        # Attention uygula
        context = torch.matmul(attention, V)
        
        # Head'leri birleştir
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class TransformerEncoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerEncoder(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_size = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Giriş projeksiyonu
        x = self.input_projection(x)
        
        # Pozisyonel kodlama
        x = self.positional_encoding(x)
        
        # Transformer katmanları
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
    def get_output_size(self) -> int:
        return self.output_size


class CNNFeatureExtractor(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list = [32, 64, 128],
        kernel_sizes: list = [50, 25, 10],
        pool_sizes: list = [8, 8, 4],
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        channels = [in_channels] + hidden_channels
        
        for i in range(len(hidden_channels)):
            layers.append(nn.Conv1d(
                channels[i], 
                channels[i+1], 
                kernel_size=kernel_sizes[i],
                padding=kernel_sizes[i] // 2
            ))
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_sizes[i]))
            layers.append(nn.Dropout(dropout))
        
        self.cnn = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        return x


class SleepTransformerEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: list = [32, 64, 128],
        cnn_kernels: list = [50, 25, 10],
        cnn_pools: list = [8, 8, 4],
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels,
            hidden_channels=cnn_channels,
            kernel_sizes=cnn_kernels,
            pool_sizes=cnn_pools,
            dropout=dropout
        )
        
        self.transformer = TransformerEncoder(
            input_size=cnn_channels[-1],
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.hidden_size = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # CNN özellik çıkarımı
        cnn_features = self.cnn(x)
        
        # Transformer encoding
        H = self.transformer(cnn_features)
        
        return H
    
    def get_output_size(self) -> int:
        return self.hidden_size


class EpochTransformer(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: list = [32, 64, 128],
        cnn_kernels: list = [50, 25, 10],
        cnn_pools: list = [8, 8, 4],
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 5,
        dropout: float = 0.1,
        pooling: str = 'cls'  # 'cls', 'mean', 'max'
    ):
        super().__init__()
        
        self.pooling = pooling
        
        self.encoder = SleepTransformerEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            cnn_pools=cnn_pools,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # CLS token
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoding
        H = self.encoder(x)  # (batch, seq_len, d_model)
        
        # Pooling
        if self.pooling == 'cls':
            # CLS token ekle ve kullan
            batch_size = H.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            H = torch.cat([cls_tokens, H], dim=1)
            pooled = H[:, 0]  # CLS token çıktısı
        elif self.pooling == 'mean':
            pooled = H.mean(dim=1)
        else:  # max
            pooled = H.max(dim=1)[0]
        
        # Sınıflandırma
        logits = self.classifier(pooled)
        
        return logits
