"""Base model class for all deep learning models."""

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, name: str = "BaseModel"):
        """
        Initialize base model.
        
        Args:
            input_shape: Shape of input data (n_segments, n_features)
            num_classes: Number of output classes
            name: Model name
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = None
    
    @abstractmethod
    def build(self) -> keras.Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled Keras model
        """
        pass
    
    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'sparse_categorical_crossentropy',
        metrics: list = None
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name
            loss: Loss function name
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy']
        
        if self.model is None:
            self.model = self.build()
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def get_model(self) -> keras.Model:
        """Get the compiled model."""
        if self.model is None:
            self.compile()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.model = self.build()
        self.model.summary()
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str) -> keras.Model:
        """Load model from file."""
        return keras.models.load_model(filepath)

