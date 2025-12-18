"""RNN-based model architectures."""

import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Flatten, Dense
from tensorflow.keras import Model
from .base import BaseModel


class RNNModel(BaseModel):
    """Simple RNN model."""

    def __init__(self, input_shape, num_classes, units=256, dropout=0.6):
        super().__init__(input_shape, num_classes, name="RNN")
        self.units = units
        self.dropout = dropout

    def build(self):
        input_tensor = Input(shape=self.input_shape)
        x = SimpleRNN(self.units, return_sequences=True)(input_tensor)
        x = SimpleRNN(self.units // 2, return_sequences=True, dropout=self.dropout)(x)
        x = Flatten()(x)
        x = Dense(64)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model


class LSTMModel(BaseModel):
    """LSTM model."""

    def __init__(self, input_shape, num_classes, units=200, dropout=0.5):
        super().__init__(input_shape, num_classes, name="LSTM")
        self.units = units
        self.dropout = dropout

    def build(self):
        input_tensor = Input(shape=self.input_shape)
        x = LSTM(self.units, return_sequences=True)(input_tensor)
        x = LSTM(self.units, return_sequences=False, dropout=self.dropout)(x)
        x = Flatten()(x)
        x = Dense(100)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor, name=self.name)
        return model
