import logging
import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import get_rnn_model

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.DEBUG)


def generate_taining_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    dataset = [[], []]

    logger.info("Now generating training data")

    encoder = LabelEncoder()
    encoder.fit(genres)

    for g in genres:

        logger.info(f"Now handling folder: {g}")

        for j, filename in enumerate(os.listdir(f'./MIR/genres/{g}')):
            songname = f'./MIR/genres/{g}/{filename}'

            data_list = []

            encoded_label = encoder.transform([g])[0]
            y_list = [encoded_label] * 10

            for i in range(10):
                sub_data_list = []
                y, sr = librosa.load(songname, mono=True, duration=3, offset=i * 3)
                sub_data_list.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.zero_crossing_rate(y)))
                sub_data_list.append(np.mean(librosa.feature.rms(y=y)))
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                for e in mfcc:
                    sub_data_list.append(np.mean(e))

                data_list.append(np.array(sub_data_list))

            dataset[0].append(np.array(data_list))
            dataset[1].append(np.array(y_list))

    dataset[0] = np.array(dataset[0])
    dataset[1] = np.array(dataset[1])

    with open('rnn_dataset.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dataset, filehandle)


def main(return_sequences=True):
    if not os.path.exists("rnn_dataset.data"):
        generate_taining_data()

    with open('rnn_dataset.data', 'rb') as filehandle:
        dataset = pickle.load(filehandle)

    X, Y = dataset[0], dataset[1]

    # normalize de dataset
    scalers = []
    for i in range(26):
        slice = X[:, :, i]
        scaler = StandardScaler()
        scaled_slice = scaler.fit_transform(slice)
        X[:, :, i] = scaled_slice
        scalers.append(scaler)

    # sla de scaler data op om tijdens inference de data op dezelfde manier te kunnen scalen
    with open('saved_scalers.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(scalers, filehandle)

    model = get_rnn_model(return_sequences=return_sequences)

    # als de RNN niet een sequence aan data returned, moeten alle labels behavle de eerste weg
    if not return_sequences:
        Y = np.delete(Y, [1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=64)

    results = model.evaluate(X_test, y_test)

    model.save('models/rnn_model.h5')


if __name__ == "__main__":
    main()
