"""TensorFlow LSTM regressor."""

from __future__ import annotations

import tensorflow as tf


def build_lstm_regressor(
    window_len: int,
    lstm_units: int = 64,
    num_lstm_layers: int = 1,
    dropout: float = 0.0,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_len, 5), name="window")
    x = inputs
    for layer_idx in range(num_lstm_layers):
        ret_seq = layer_idx < num_lstm_layers - 1
        x = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=ret_seq,
            name=f"lstm_{layer_idx}",
        )(x)
        if dropout > 0 and layer_idx == num_lstm_layers - 1:
            x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="y_hat")(x)
    model = tf.keras.Model(inputs, out, name="lstm_frequency_selector")
    return model
