import logging
import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import get_rnn_model
from utils.feature_extraction import extract_features_for_audio_clip

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.DEBUG)


def generate_training_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    dataset = [[], []]

    logger.info("Now generating training data")

    encoder = LabelEncoder()
    encoder.fit(genres)

    for g in genres:

        logger.info(f"Now handling folder: {g}")

        for filename in os.listdir(f'./dataset/{g}'):
            songname = f'./dataset/{g}/{filename}'

            data_list = []

            encoded_label = encoder.transform([g])[0]
            y_list = [encoded_label] * 10

            # Our audio clips are 30 seconds long,
            # generate sequences of 10 inputs and outputs for the RNN input.
            # This means our input is 10 3-second clips,
            # and our output will be 10 predictions, or just one if "return_sequences" is disabled in the lstm.
            for i in range(10):
                y, sr = librosa.load(songname, mono=True, duration=3, offset=i * 3)

                data_list.append(extract_features_for_audio_clip(y=y, sr=sr))

            dataset[0].append(data_list)
            dataset[1].append(y_list)

    dataset[0] = np.array(dataset[0])
    dataset[1] = np.array(dataset[1])

    with open('processed_datasets/rnn_dataset.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dataset, filehandle)


def main(return_sequences=True):
    if not os.path.exists("processed_datasets/rnn_dataset.data"):
        generate_training_data()

    with open('processed_datasets/rnn_dataset.data', 'rb') as filehandle:
        dataset = pickle.load(filehandle)

    X, Y = dataset[0], dataset[1]

    model = get_rnn_model(return_sequences=return_sequences, stateful=False)
    model.layers[0].adapt(X)

    # If the LSTM is set to not return a whole sequence, but just 1 output, we need to have just a single label
    if not return_sequences:
        Y = np.delete(Y, [1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model.fit(X_train,
              y_train,
              epochs=100,
              batch_size=64)

    model.evaluate(X_test, y_test)

    model.save('saved_models/rnn_model.h5')


if __name__ == "__main__":
    main()
