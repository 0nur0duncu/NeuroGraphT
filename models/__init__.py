"""
Models modülü için __init__.py
Sleep Stage Classification with CNN-Transformer + GCN
"""

from .cnn_module import CNNBlock, CNNEncoder
from .transformer_module import (
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
    CNNFeatureExtractor,
    SleepTransformerEncoder,
    EpochTransformer
)
from .gcn_module import GCNClassifier
from .epigraphnet import EpiGraphNet, CNNTransformerEncoder, GraphBuilder, GCNEncoder
from .baselines import BaselineCNNTransformer, Baseline1DCNNTransformer

# Backward compatibility
from .baselines import BaselineCNNLSTM, Baseline1DCNNLSTM

__all__ = [
    # CNN
    "CNNBlock",
    "CNNEncoder",
    # Transformer
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "CNNFeatureExtractor",
    "SleepTransformerEncoder",
    "EpochTransformer",
    # GCN
    "GCNClassifier",
    # Main Models
    "EpiGraphNet",
    "CNNTransformerEncoder",
    "GraphBuilder",
    "GCNEncoder",
    # Baselines
    "BaselineCNNTransformer",
    "Baseline1DCNNTransformer",
    # Backward compatibility
    "BaselineCNNLSTM",
    "Baseline1DCNNLSTM",
]
