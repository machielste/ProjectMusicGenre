import os
import librosa

import thirdparty.utils as utils  # note my fix to fma utils, https://github.com/mdeff/fma/issues/34

AUDIO_DIR = 'fma_medium/'
tracks = utils.load('tracks.csv')

base_dir = "Samples_medium/"

os.mkdir(base_dir[:-1])

small = tracks['set', 'subset'] <= 'medium'

y_small = tracks.loc[small, ('track', 'genre_top')]

total_errors = 0
for track_id, genre in y_small.iteritems():
    if not os.path.exists(base_dir + genre):
        os.mkdir(base_dir + genre)

    mp3_filename = utils.get_audio_path(AUDIO_DIR, track_id)
    out_wav_filename = base_dir + genre + '/' + str(track_id) + '.wav'

    try:
        print("reading ", mp3_filename)

        data_list = []

        encoded_label = encoder.transform([g])[0]
        y_list = [encoded_label] * 10

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

        dataset[0].append(np.array(data_list))
        dataset[1].append(np.array(y_list))



    except:
        total_errors += 1

print(str(total_errors))
