from tkinter import Tk
from tkinter.filedialog import askopenfilename


def let_user_select_file():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    return filepath
