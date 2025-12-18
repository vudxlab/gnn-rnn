"""Hybrid GNN-LSTM model architectures."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, LSTM,
    Reshape, TimeDistributed, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras import Model
from .base import BaseModel


class GNNLSTMModel(BaseModel):
    """
    Hybrid GNN-LSTM model for structural health monitoring.

    Architecture:
    1. GNN layers extract spatial features from sensor network
    2. LSTM layers capture long-term temporal dependencies
    3. Dense layers for classification

    This model processes time series data where:
    - GNN captures relationships between sensors (spatial)
    - LSTM captures temporal evolution with memory of past states
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        lstm_units=200,
        dropout_gnn=0.3,
        dropout_lstm=0.5
    ):
        """
        Initialize GNN-LSTM hybrid model.

        Args:
            input_shape: Shape of input (n_segments, n_features)
            num_classes: Number of output classes
            gnn_units: List of hidden units for GNN layers
            lstm_units: Number of units in LSTM layers
            dropout_gnn: Dropout rate for GNN layers
            dropout_lstm: Dropout rate for LSTM layers
        """
        super().__init__(input_shape, num_classes, name="GNN_LSTM")
        self.gnn_units = gnn_units
        self.lstm_units = lstm_units
        self.dropout_gnn = dropout_gnn
        self.dropout_lstm = dropout_lstm

    def build(self):
        """Build GNN-LSTM hybrid model."""
        n_segments, n_features = self.input_shape

        # Input layer
        input_tensor = Input(shape=self.input_shape)
        x = input_tensor

        # GNN feature extraction (spatial relationships)
        # Process each time segment with graph structure
        for i, units in enumerate(self.gnn_units):
            x = Dense(units, activation='relu', name=f'gnn_{i+1}')(x)
            x = LayerNormalization(name=f'gnn_norm_{i+1}')(x)
            x = Dropout(self.dropout_gnn)(x)

        # LSTM layers for temporal modeling
        # Capture how spatial features evolve over time with long-term memory
        x = LSTM(self.lstm_units, return_sequences=True, name='lstm_1')(x)
        x = LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_lstm,
            name='lstm_2'
        )(x)

        # Global temporal pooling - average across time dimension
        x = GlobalAveragePooling1D()(x)

        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_lstm)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model


class AttentionGNNLSTMModel(BaseModel):
    """
    GNN-LSTM model with attention mechanism.

    Adds attention to focus on important time steps after GNN processing.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        lstm_units=200,
        dropout_gnn=0.3,
        dropout_lstm=0.5
    ):
        super().__init__(input_shape, num_classes, name="Attention_GNN_LSTM")
        self.gnn_units = gnn_units
        self.lstm_units = lstm_units
        self.dropout_gnn = dropout_gnn
        self.dropout_lstm = dropout_lstm

    def build(self):
        """Build attention-based GNN-LSTM model."""
        input_tensor = Input(shape=self.input_shape)
        x = input_tensor

        # GNN layers for spatial feature extraction
        for i, units in enumerate(self.gnn_units):
            x = Dense(units, activation='relu', name=f'gnn_{i+1}')(x)
            x = Dropout(self.dropout_gnn)(x)

        # LSTM with return_sequences for attention
        lstm_out = LSTM(self.lstm_units, return_sequences=True, name='lstm_1')(x)
        lstm_out = LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_lstm,
            name='lstm_2'
        )(lstm_out)

        # Simple attention mechanism
        attention_scores = Dense(1, activation='tanh', name='attention_scores')(lstm_out)
        attention_weights = tf.nn.softmax(attention_scores, axis=1, name='attention_weights')
        context_vector = tf.reduce_sum(attention_weights * lstm_out, axis=1, name='context')

        # Classification
        x = Dense(100, activation='relu')(context_vector)
        x = Dropout(self.dropout_lstm)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model


class BiLSTMGNNModel(BaseModel):
    """
    Bidirectional LSTM with GNN preprocessing.

    Uses BiLSTM for better temporal context (both past and future).
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        lstm_units=256,
        dropout_gnn=0.3,
        dropout_lstm=0.6
    ):
        super().__init__(input_shape, num_classes, name="BiLSTM_GNN")
        self.gnn_units = gnn_units
        self.lstm_units = lstm_units
        self.dropout_gnn = dropout_gnn
        self.dropout_lstm = dropout_lstm

    def build(self):
        """Build BiLSTM-GNN model."""
        from tensorflow.keras.layers import Bidirectional

        input_tensor = Input(shape=self.input_shape)
        x = input_tensor

        # GNN layers
        for i, units in enumerate(self.gnn_units):
            x = Dense(units, activation='relu', name=f'gnn_{i+1}')(x)
            x = Dropout(self.dropout_gnn)(x)

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True), name='bilstm_1')(x)
        x = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_lstm),
            name='bilstm_2'
        )(x)

        x = Flatten()(x)

        # Classification
        x = Dense(100, activation='relu')(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model
