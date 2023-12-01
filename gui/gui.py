import tkinter as tk
from typing import List

import vlc

from deeplearning.infer import RnnInference
from utils.tkinter_util import let_user_select_file


class Gui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Music genre recognizer")

        button = tk.Button(self.root, text="Select an audio file", fg="purple", command=self.receive_user_sound_file,
                           width=22)
        button.grid(row=0, column=0)
        button2 = tk.Button(self.root, text="Play and analyze", fg="green", command=self.play, width=22)
        button2.grid(row=0, column=1)
        button3 = tk.Button(self.root, text="Stop", fg="red", command=self.stop, width=22)
        button3.grid(row=0, column=2)

        self.label = tk.Label(self.root, text="Currently selected file: None")
        self.label.grid(row=2, column=1)

        self.text = tk.Text(self.root)
        self.text.insert('1.0', "Select audio file")
        self.text.grid(row=3, column=1)

        self.root.grid()
        self.root.protocol("WM_DELETE_WINDOW", self._delete_window)

        self.media_player = None
        self.current_sound_file = None
        self.current_after_job = None
        self.prediction = None
        self.prediction_index = 0

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
            self.popup_message("Please select an MP3 or a WAV file")

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

        self.update_selected_file_label()

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

    def draw_prediction(self, full_prediction: List):
        self.clear_text_window()

        for single_prediction in full_prediction:
            single_prediction[1] = round(single_prediction[1] * 100)
            to_insert = f'{single_prediction[0]} {single_prediction[1]}%'
            to_insert = self.remove_garbage_from_string(to_insert)
            self.text.insert('end', to_insert + '\n')

    def clear_prediction_field(self):
        self.clear_text_window()
        self.text.insert('1.0', "Please select an audio file")

    def update_selected_file_label(self):
        if self.current_sound_file:
            split = self.current_sound_file.split("/")
            new_text = f"Currently selected audio file: {split[len(split) - 1]}"
            self.label.config(text=new_text)
        else:
            self.label.config(text="Currently selected file: None")

    def clear_text_window(self):
        self.text.delete('1.0', '10.0')

    def _delete_window(self):
        self.root.destroy()

    @staticmethod
    def get_selected_file():
        return let_user_select_file()

    @staticmethod
    def popup_message(message_to_display: str):
        NORM_FONT = ("Helvetica", 10)
        popup = tk.Tk()
        popup.wm_title("!")
        label = tk.Label(popup, text=message_to_display, font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = tk.Button(popup, text="Ok", command=popup.destroy)
        B1.pack()
        popup.mainloop()

    @staticmethod
    def remove_garbage_from_string(string: str):
        string = string.translate({ord(c): None for c in r"[]\/"})
        return string
