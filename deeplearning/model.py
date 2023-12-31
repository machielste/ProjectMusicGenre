import tensorflow as tf
from keras import Model


def get_rnn_model(return_sequences, stateful, classes=10) -> Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Normalization(axis=2),
        tf.keras.layers.LSTM(1024, return_sequences=return_sequences,
                             dropout=0.1,
                             recurrent_dropout=0,
                             stateful=stateful
                             ),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(classes, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
