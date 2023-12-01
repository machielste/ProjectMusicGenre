# Simple python GUI application which analyzes music, and predicts the genre while the song plays.

## Environment requirements:

- Installation of requirements.txt
- Cuda 11.2
- Cudnn 8.1 or higher
- Ffmpeg installation
- VLC Media Player installation

Depending on the user's system, only requirements.txt needs to be installed, as it contains the packaged mentioned below
it.
The user can replace the tensorflow-gpu package with regular tensorflow if no GPU is available.

## Usage:

This project provides code to train a model on the GTZAN dataset, and a GUI which uses the model to analyze songs.
A pre-trained model file is provided in this repo, thus the user does not need to train the model by themselves.

Run the following command while in a valid python environment:

```
python main.py
```

Alternatively you can do the following to generate the dataset and model by yourself:

- Remove rnn_model.h5 and rnn_dataset.data.
- Provide the GTZAN dataset in a folder called "dataset" under the "deeplearning" folder.
- Run "train_rnn_model.py"

This wil first generate the dataset, and then proceed to train the model, after which the dataset and model are saved.