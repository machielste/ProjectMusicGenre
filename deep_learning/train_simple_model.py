import csv
import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import get_simple_model


def generate_csv():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        for filename in os.listdir(f'./MIR/genres/{g}'):
            songname = f'./MIR/genres/{g}/{filename}'

            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)}' \
                        f' {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
            writer.writerow(to_append.split())


def main():
    if not os.path.exists("data.csv"):
        generate_csv()

    data = pd.read_csv('data.csv')
    data.head()
    data = data.drop(['filename'], axis=1)

    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = get_simple_model(X_train.shape[1])

    history = model.fit(X_train,
                        y_train,
                        epochs=20,
                        batch_size=128)

    results = model.evaluate(X_test, y_test)

    model.save('models/simple_model.h5')


if __name__ == "__main__":
    main()
