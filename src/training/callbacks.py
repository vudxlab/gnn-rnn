"""Training callbacks."""

from tensorflow import keras
from typing import List


def get_callbacks(
    patience: int = 50,
    min_delta: float = 0.00001,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    lr_min_delta: float = 0.0001,
    cooldown: int = 5
) -> List[keras.callbacks.Callback]:
    """
    Get list of training callbacks.
    
    Args:
        patience: Number of epochs with no improvement for early stopping
        min_delta: Minimum change to qualify as improvement
        lr_patience: Number of epochs with no improvement before reducing LR
        lr_factor: Factor by which learning rate will be reduced
        lr_min_delta: Minimum change for learning rate reduction
        cooldown: Number of epochs to wait before resuming normal operation
    
    Returns:
        List of Keras callbacks
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=1,
        mode="min",
        restore_best_weights=True
    )
    
    lr_reducer = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_factor,
        patience=lr_patience,
        min_delta=lr_min_delta,
        cooldown=cooldown,
        verbose=0
    )
    
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    return [early_stopping, lr_reducer, terminate_on_nan]

