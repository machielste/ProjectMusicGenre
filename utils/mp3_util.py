from mutagen.mp3 import MP3


def get_song_runtime(filename) -> int:
    audio = MP3(filename)
    length = audio.info.length
    return int(length)
