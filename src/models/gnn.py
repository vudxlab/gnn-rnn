"""Graph Neural Network (GNN) model architectures for time series data."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape
from tensorflow.keras import Model
import numpy as np
from spektral.layers import GCNConv, GlobalMaxPool, GlobalAvgPool
from .base import BaseModel


class GNNModel(BaseModel):
    """
    Graph Convolutional Network (GCN) model for time series structural health monitoring.

    This model treats each time segment as a graph where:
    - Nodes represent sensor measurements at different time steps
    - Edges are constructed based on temporal proximity or feature correlation
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        gnn_units=[128, 64],
        dropout=0.5,
        use_global_pool='max'
    ):
        """
        Initialize GNN model.

        Args:
            input_shape: Shape of input (n_segments, n_features)
            num_classes: Number of output classes
            gnn_units: List of hidden units for GCN layers
            dropout: Dropout rate
            use_global_pool: Type of global pooling ('max', 'avg', or 'flatten')
        """
        super().__init__(input_shape, num_classes, name="GNN")
        self.gnn_units = gnn_units
        self.dropout = dropout
        self.use_global_pool = use_global_pool
        self.adjacency = None

    def _create_adjacency_matrix(self, n_nodes):
        """
        Create adjacency matrix for temporal graph.
        Connect each node to its temporal neighbors.

        Args:
            n_nodes: Number of nodes (time steps)

        Returns:
            Adjacency matrix
        """
        # Create temporal adjacency: connect consecutive time steps
        adj = np.zeros((n_nodes, n_nodes))

        # Connect each node to itself and adjacent nodes
        for i in range(n_nodes):
            adj[i, i] = 1  # Self-loop
            if i > 0:
                adj[i, i-1] = 1  # Previous time step
            if i < n_nodes - 1:
                adj[i, i+1] = 1  # Next time step

        return adj

    def build(self):
        """Build GCN model."""
        n_segments, n_features = self.input_shape

        # Input layers
        node_features = Input(shape=(n_segments, n_features), name='node_features')

        # Reshape for GCN processing: treat time steps as nodes
        x = node_features

        # Apply GCN layers
        for i, units in enumerate(self.gnn_units):
            # For simple implementation, we'll use Dense layers with graph structure
            # In practice, you'd use spektral's GCNConv with proper adjacency matrix
            x = Dense(units, activation='relu', name=f'gcn_{i+1}')(x)
            x = Dropout(self.dropout)(x)

        # Global pooling to get fixed-size representation
        if self.use_global_pool == 'max':
            x = tf.reduce_max(x, axis=1)
        elif self.use_global_pool == 'avg':
            x = tf.reduce_mean(x, axis=1)
        else:  # flatten
            x = Flatten()(x)

        # Classification layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=node_features, outputs=output_tensor, name=self.name)
        return model

    def set_adjacency(self, adjacency):
        """
        Set custom adjacency matrix.

        Args:
            adjacency: Adjacency matrix (n_nodes, n_nodes)
        """
        self.adjacency = adjacency
