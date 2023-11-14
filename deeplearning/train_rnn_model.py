import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import get_rnn_model
from utils.constants import GENRES
from utils.feature_extraction import extract_features_for_audio_clip


def generate_training_data():
    dataset = [[], []]
    encoder = LabelEncoder()
    encoder.fit(GENRES)

    for g in GENRES:
        for filename in os.listdir(f'./dataset/{g}'):
            song_filepath = f'./dataset/{g}/{filename}'

            encoded_label = encoder.transform([g])[0]
            y_list = [encoded_label] * 10

            # Our audio clips are 30 seconds long,
            # generate sequences of 10 inputs and outputs for the RNN input.
            # This means our input is 10 3-second clips,
            # and our output will be 10 predictions.
            librosa_features = []
            for i in range(10):
                y, sr = librosa.load(song_filepath, mono=True, duration=3, offset=i * 3)

                librosa_features.append(extract_features_for_audio_clip(y=y, sr=sr))

            dataset[0].append(librosa_features)
            dataset[1].append(y_list)

    dataset[0] = np.array(dataset[0])
    dataset[1] = np.array(dataset[1])

    with open('processed_datasets/rnn_dataset.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dataset, filehandle)


def main():
    if not os.path.exists("processed_datasets/rnn_dataset.data"):
        generate_training_data()

    with open('processed_datasets/rnn_dataset.data', 'rb') as filehandle:
        dataset = pickle.load(filehandle)

    X, Y = dataset[0], dataset[1]

    model = get_rnn_model(return_sequences=True, stateful=False)
    # use training data to initialize normalization layer of model
    model.layers[0].adapt(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model.fit(X_train,
              y_train,
              epochs=100,
              batch_size=64)

    model.evaluate(X_test, y_test)

    model.save('saved_models/rnn_model.h5')


if __name__ == "__main__":
    main()
