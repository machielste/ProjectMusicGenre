import logging
import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import thirdparty.utils as utils
from deep_learning.model import get_rnn_model

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.DEBUG)


def generate_taining_data():
    AUDIO_DIR = '.fma/fma_medium'

    tracks = utils.load("./fma/tracks.csv")

    medium = tracks['set', 'subset'] <= 'medium'
    y_small = tracks.loc[medium, ('track', 'genre_top')]

    CLASSES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
               'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
               'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

    dataset = [[], []]

    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    total_errors = 0
    for count, (track_id, genre) in enumerate(y_small.iteritems()):

        mp3_filename = utils.get_audio_path(AUDIO_DIR, track_id)[1:]

        data_list = []

        encoded_label = encoder.transform([genre])[0]
        y_list = [encoded_label] * 10

        logger.info("Now handling {}".format(mp3_filename))

        try:
            for i in range(10):
                sub_data_list = []
                y, sr = librosa.load(mp3_filename, mono=True, duration=3, offset=i * 3)
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

        except:
            logger.info("error occured in file {}".format(mp3_filename))
            total_errors += 1
            continue

        dataset[0].append(np.array(data_list))
        dataset[1].append(np.array(y_list))

    dataset[0] = np.array(dataset[0])
    dataset[1] = np.array(dataset[1])

    logger.info(str(total_errors))

    with open('rnn_dataset_fma.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dataset, filehandle)


def main(return_sequences=True):
    if not os.path.exists("rnn_dataset_fma.data"):
        generate_taining_data()

    with open('rnn_dataset_fma.data', 'rb') as filehandle:
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
    with open('saved_scalers_fma.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(scalers, filehandle)

    model = get_rnn_model(return_sequences=return_sequences, classes=16)

    # als de RNN niet een sequence aan data returned, moeten alle labels behavle de eerste weg
    if not return_sequences:
        Y = np.delete(Y, [1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=64)

    results = model.evaluate(X_test, y_test)

    model.save('models/rnn_model_fma.h5')


if __name__ == "__main__":
    main()
