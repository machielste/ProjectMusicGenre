import logging
import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

        for filename in os.listdir(f'./dataset/{g}'):
            songname = f'./dataset/{g}/{filename}'

            data_list = []

            encoded_label = encoder.transform([g])[0]
            y_list = [encoded_label] * 10

            # our audio clips are 30 seconds long, generate an input output pair for every 3 seconds
            for i in range(10):
                sub_data_list = []
                logger.info(f"now processing {songname}")

                y, sr = librosa.load(songname, mono=True, duration=3, offset=i * 3)

                sub_data_list.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                sub_data_list.append(np.mean(librosa.feature.zero_crossing_rate(y)))
                sub_data_list.append(np.mean(librosa.feature.rms(y=y)))
                # mfcc = librosa.feature.mfcc(y=y, sr=sr)
                # for e in mfcc:
                #     sub_data_list.append(np.mean(e))
                #
                data_list.append(sub_data_list)

            dataset[0].append(data_list)
            dataset[1].append(y_list)

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

    model.save('models/rnn_model.keras')


if __name__ == "__main__":
    main()
