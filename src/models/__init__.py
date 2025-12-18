"""Deep learning model architectures for structural health monitoring."""

from .base import BaseModel

# RNN-based models
from .rnn import RNNModel, LSTMModel

# GNN-based models
from .gnn import GNNModel

# Hybrid GNN-RNN models
from .gnn_rnn import GNNRNNModel, AttentionGNNRNNModel

# Hybrid GNN-LSTM models
from .gnn_lstm import (
    GNNLSTMModel,
    AttentionGNNLSTMModel,
    BiLSTMGNNModel
)

__all__ = [
    # Base
    'BaseModel',

    # RNN models
    'RNNModel',
    'LSTMModel',

    # GNN models
    'GNNModel',

    # GNN-RNN hybrids
    'GNNRNNModel',
    'AttentionGNNRNNModel',

    # GNN-LSTM hybrids
    'GNNLSTMModel',
    'AttentionGNNLSTMModel',
    'BiLSTMGNNModel',
]
