import pickle

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.mp3_wav_util import get_wav_length


class SimpleInference:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            "deep_learning/models/simple_model.h5")

    def infer(self, path_to_file):
        # filename,chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20,label

        y, sr = librosa.load(path_to_file, mono=True)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        input = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
                 np.mean(zcr)]

        for e in mfcc:
            input.append(np.mean(e))

        data = pd.read_csv("deep_learning/data.csv")
        data.head()
        data = data.drop(['filename'], axis=1)

        scaler = StandardScaler()
        scaler.fit(np.array(data.iloc[:, :-1], dtype=float))

        genre_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        encoder.fit(genre_list)

        arr = np.array(input, dtype=float)
        arr = np.expand_dims(arr, 0)
        X = scaler.transform(arr)

        prediction_list = self.model.predict(X)
        prediction_list = np.squeeze(prediction_list)

        final_result = []
        for i in range(0, 3):
            res = np.argmax(prediction_list)
            final_result.append([encoder.inverse_transform([res]), prediction_list[res]])
            prediction_list[res] = 0
        return final_result


class RnnInference:
    def __init__(self, fma=False):
        if fma:
            self.model = tf.keras.models.load_model(
                "deep_learning/models/rnn_model_fma.h5")
        else:
            self.model = tf.keras.models.load_model(
                "deep_learning/models/rnn_model.h5")

        self.fma = fma

    def infer(self, path_to_file):
        data_list = []

        length = get_wav_length(path_to_file)
        length = length - length % 3

        # genereer features voor elke 3 seconden in het liedje
        for i in range(int(length / 3)):
            sub_data_list = []
            y, sr = librosa.load(path_to_file, mono=True, duration=3, offset=i * 3)
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

        data_list = np.array(data_list)

        if self.fma:
            with open('deep_learning/saved_scalers_fma.data', 'rb') as filehandle:
                scaler_params = pickle.load(filehandle)
        else:
            with open('deep_learning/saved_scalers.data', 'rb') as filehandle:
                scaler_params = pickle.load(filehandle)

        data_list = np.expand_dims(data_list, axis=0)

        for i in range(26):
            scaler = scaler_params[i]
            slice = data_list[:, :, i]

            rounded_time_series_length = np.shape(slice)[1] - np.shape(slice)[1] % 10

            for g in range(0, rounded_time_series_length, 10):
                slice[:, g:g + 10] = scaler.transform(slice[:, g:g + 10])

            slice[:, -10:] = scaler.transform(slice[:, -10:])

            data_list[:, :, i] = slice

        prediction_list = self.model.predict(data_list)
        prediction_list = np.squeeze(prediction_list)

        genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        encoder = LabelEncoder()
        encoder.fit(genres)

        final_result = []

        for pred in prediction_list:
            current_result = []
            for i in range(3):
                res = np.argmax(pred)
                current_result.append([encoder.inverse_transform([res]), pred[res]])
                pred[res] = 0

            final_result.append(current_result)

        return final_result
