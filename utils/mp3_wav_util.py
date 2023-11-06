from mutagen.mp3 import MP3


def get_song_runtime(fname):
    audio = MP3(fname)
    length = audio.info.length
    return int(length)
