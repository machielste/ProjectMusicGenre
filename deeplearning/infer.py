import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from utils.feature_extraction import extract_features_for_audio_clip
from utils.mp3_util import get_song_runtime


class RnnInference:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            "deeplearning/saved_models/rnn_model.h5")
        self.model.stateful = True
        self.model.reset_states()

    def infer(self, path_to_file):
        data_list = []

        length = get_song_runtime(path_to_file)

        for i in range(int(length / 3)):
            y, sr = librosa.load(path_to_file, mono=True, duration=3, offset=i * 3)
            data_list.append(np.array(extract_features_for_audio_clip(y=y, sr=sr)))

        # data_list = np.expand_dims(data_list, axis=0)
        data_list = np.array(data_list)

        prediction_list = []
        for item in data_list:
            prediction_list.append(self.model.predict(item))
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
