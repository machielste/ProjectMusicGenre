import tensorflow as tf


def get_rnn_model(return_sequences=True, classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(1024, return_sequences=return_sequences,
                             dropout=0.4,
                             recurrent_dropout=0.0,
                             ),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(classes, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
