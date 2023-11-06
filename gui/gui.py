from tkinter import *

import vlc

from deeplearning.infer import RnnInference
from utils.upload_sound_file import let_user_select_file


class Gui:
    def __init__(self):
        self.root = Tk()
        self.root.title("Music genre recognizer")

        button = Button(self.root, text="Select an audio file", fg="purple", command=self.receive_user_sound_file,
                        width=22)
        button.grid(row=0, column=0)
        button2 = Button(self.root, text="Play and analyze", fg="green", command=self.play, width=22)
        button2.grid(row=0, column=1)
        button3 = Button(self.root, text="Stop", fg="red", command=self.stop, width=22)
        button3.grid(row=0, column=2)

        self.label = Label(self.root, text="Currently selected file: None")
        self.label.grid(row=2, column=1)

        self.text = Text(self.root)
        self.text.insert('1.0', "Select audio file")
        self.text.grid(row=3, column=1)

        self.root.grid()
        self.root.protocol("WM_DELETE_WINDOW", self._delete_window)

        # init for vars
        self.media_player = None
        self.current_sound_file = None
        self.current_after_job = None
        self.prediction = None
        self.prediction_index = 0

        # init for inference class
        self.inference = RnnInference()

    def run(self):
        self.root.mainloop()

    def receive_user_sound_file(self):
        self.stop()

        user_selected_file = let_user_select_file()

        if user_selected_file.endswith("mp3") or user_selected_file.endswith("wav"):
            self.current_sound_file = user_selected_file
            self.update_selected_file_label()

        else:
            self.popupmsg("Please select an MP3 or a WAV file")

    def play(self):
        if not self.media_player and self.current_sound_file is not None:
            self.predict_and_show()
            self.media_player = vlc.MediaPlayer(self.current_sound_file)
            self.media_player.play()

    def stop(self, clear_prediction_field=True):
        if self.media_player:
            self.media_player.stop()
            self.media_player = None
        self.current_sound_file = None
        self.prediction_index = 0

        try:
            self.root.after_cancel(self.current_after_job)
        except ValueError:
            pass

        if clear_prediction_field:
            self.clear_prediction_field()

    def predict_and_show(self):
        self.prediction = self.inference.infer(self.current_sound_file)
        self.continuously_update_predict_window()

    def continuously_update_predict_window(self):
        # We predicted a result for each 3 second bit of the song, show the results while the song is playing
        if self.prediction_index < len(self.prediction):
            self.draw_prediction(self.prediction[self.prediction_index])
            self.prediction_index += 1
            if self.current_sound_file is not None:
                self.current_after_job = self.root.after(3000, self.continuously_update_predict_window)
        else:
            self.stop(clear_prediction_field=False)

    def draw_prediction(self, pred):
        self.text.delete('1.0', '10.0')

        for x in pred:
            to_insert = (str(x[0]) + " " + str(x[1]))
            to_insert = self.remove_garbage_from_string(to_insert)
            self.text.insert('end', to_insert + '\n')

    def clear_prediction_field(self):
        self.text.delete('1.0', '10.0')
        self.text.insert('1.0', "Please select an audio file")

    def update_selected_file_label(self):
        split = self.current_sound_file.split("/")
        new_text = "Currently selected audio file: {}".format(split[len(split) - 1])
        self.label.config(text=new_text)

    def _delete_window(self):
        self.root.destroy()

    @staticmethod
    def get_selected_file():
        return let_user_select_file()

    @staticmethod
    def popupmsg(msg):
        NORM_FONT = ("Helvetica", 10)
        popup = Tk()
        popup.wm_title("!")
        label = Label(popup, text=msg, font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = Button(popup, text="Ok", command=popup.destroy)
        B1.pack()
        popup.mainloop()

    @staticmethod
    def remove_garbage_from_string(string):
        string = string.translate({ord(c): None for c in r"[]\/"})
        return string
