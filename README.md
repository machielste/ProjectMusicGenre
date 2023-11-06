Simple python GUI application which analyzes music, and predicts the genre while the song plays.

Environment requirements:

- Cuda 11.2
- Cudnn 8.1 or higher
- Ffmpeg installation
- Vlc installation
- Installation of requirements.txt

Alternatively you can replace the tensorflow-gpu package with regular tensorflow.

Usage:

- run "main.py"

Alternatively you can do the following to generate the dataset and model by yourself:

- remove rnn_model.h5 and rnn_dataset.data
- provide the GTZAN dataset in a folder called "dataset" under the "deeplearning" folder
- run "train_rnn_model.py"