import librosa
import numpy as np


def extract_features_for_audio_clip(y, sr):
    sub_data_list = []
    sub_data_list.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    sub_data_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    sub_data_list.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    sub_data_list.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    sub_data_list.append(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    sub_data_list.append(np.mean(librosa.feature.rms(y=y)))
    sub_data_list.append(np.mean(librosa.feature.tempo(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for e in mfcc:
        sub_data_list.append(np.mean(e))

    return sub_data_list
