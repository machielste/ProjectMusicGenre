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
        length = get_song_runtime(path_to_file)
        data_list = []

        for i in range(int(length / 3)):
            y, sr = librosa.load(path_to_file, mono=True, duration=3, offset=i * 3)
            data_list.append(extract_features_for_audio_clip(y=y, sr=sr))

        prediction_list = np.squeeze([self.model.predict(item) for item in np.array(data_list)])

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
