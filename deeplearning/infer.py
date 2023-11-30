from typing import List

import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from utils.constants import GENRES
from utils.feature_extraction import extract_features_for_audio_clip
from utils.mp3_util import get_song_runtime


class RnnInference:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            "deeplearning/saved_models/rnn_model.h5")
        self.model.stateful = True
        self.model.reset_states()

    def infer(self, song_path) -> List:
        """
        :param song_path: path of song to run inference on
        :return: list of predictions in ["genre", "confidence"] pairs
        """
        length = get_song_runtime(song_path)

        extracted_features = []
        for i in range(int(length / 3)):
            y, sr = librosa.load(song_path, mono=True, duration=3, offset=i * 3)
            extracted_features.append(extract_features_for_audio_clip(y=y, sr=sr))

        prediction_list = np.squeeze([self.model.predict(item) for item in np.array(extracted_features)])

        encoder = LabelEncoder()
        encoder.fit(GENRES)

        top_predictions = []
        for prediction in prediction_list:
            current_result = []
            for i in range(3):
                res = np.argmax(prediction)
                current_result.append([encoder.inverse_transform([res]), prediction[res]])
                prediction[res] = 0

            top_predictions.append(current_result)

        return top_predictions
