"""Hybrid GNN-RNN model architectures."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, SimpleRNN,
    Reshape, TimeDistributed
)
from tensorflow.keras import Model
from .base import BaseModel


class GNNRNNModel(BaseModel):
    """
    Hybrid GNN-RNN model for structural health monitoring.

    Architecture:
    1. GNN layers extract spatial features from sensor network
    2. RNN layers capture temporal dependencies
    3. Dense layers for classification

    This model processes time series data where:
    - GNN captures relationships between sensors (spatial)
    - RNN captures temporal evolution of patterns
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        rnn_units=256,
        dropout_gnn=0.3,
        dropout_rnn=0.6
    ):
        """
        Initialize GNN-RNN hybrid model.

        Args:
            input_shape: Shape of input (n_segments, n_features)
            num_classes: Number of output classes
            gnn_units: List of hidden units for GNN layers
            rnn_units: Number of units in RNN layers
            dropout_gnn: Dropout rate for GNN layers
            dropout_rnn: Dropout rate for RNN layers
        """
        super().__init__(input_shape, num_classes, name="GNN_RNN")
        self.gnn_units = gnn_units
        self.rnn_units = rnn_units
        self.dropout_gnn = dropout_gnn
        self.dropout_rnn = dropout_rnn

    def build(self):
        """Build GNN-RNN hybrid model."""
        n_segments, n_features = self.input_shape

        # Input layer
        input_tensor = Input(shape=self.input_shape)
        x = input_tensor

        # GNN feature extraction (spatial relationships)
        # Process each time segment with graph structure
        for i, units in enumerate(self.gnn_units):
            x = Dense(units, activation='relu', name=f'gnn_{i+1}')(x)
            x = Dropout(self.dropout_gnn)(x)

        # RNN layers for temporal modeling
        # Capture how spatial features evolve over time
        x = SimpleRNN(self.rnn_units, return_sequences=True, name='rnn_1')(x)
        x = SimpleRNN(
            self.rnn_units // 2,
            return_sequences=True,
            dropout=self.dropout_rnn,
            name='rnn_2'
        )(x)

        # Global temporal pooling
        x = Flatten()(x)

        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rnn)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model


class AttentionGNNRNNModel(BaseModel):
    """
    GNN-RNN model with attention mechanism.

    Adds attention to focus on important time steps after GNN processing.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        rnn_units=256,
        dropout_gnn=0.3,
        dropout_rnn=0.6
    ):
        super().__init__(input_shape, num_classes, name="Attention_GNN_RNN")
        self.gnn_units = gnn_units
        self.rnn_units = rnn_units
        self.dropout_gnn = dropout_gnn
        self.dropout_rnn = dropout_rnn

    def build(self):
        """Build attention-based GNN-RNN model."""
        input_tensor = Input(shape=self.input_shape)
        x = input_tensor

        # GNN layers
        for i, units in enumerate(self.gnn_units):
            x = Dense(units, activation='relu', name=f'gnn_{i+1}')(x)
            x = Dropout(self.dropout_gnn)(x)

        # RNN with return_sequences for attention
        rnn_out = SimpleRNN(self.rnn_units, return_sequences=True, name='rnn_1')(x)
        rnn_out = SimpleRNN(
            self.rnn_units // 2,
            return_sequences=True,
            dropout=self.dropout_rnn,
            name='rnn_2'
        )(rnn_out)

        # Simple attention mechanism
        attention_scores = Dense(1, activation='tanh', name='attention_scores')(rnn_out)
        attention_weights = tf.nn.softmax(attention_scores, axis=1, name='attention_weights')
        context_vector = tf.reduce_sum(attention_weights * rnn_out, axis=1, name='context')

        # Classification
        x = Dense(128, activation='relu')(context_vector)
        x = Dropout(self.dropout_rnn)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model
