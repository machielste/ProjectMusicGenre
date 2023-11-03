import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from utils.mp3_wav_util import get_wav_length


class RnnInference:
    def __init__(self):

        self.model = tf.keras.models.load_model(
            "deeplearning/saved_models/rnn_model.keras")

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

            data_list.append(np.array(sub_data_list))

        data_list = np.array(data_list)
        data_list = np.expand_dims(data_list, axis=0)

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
