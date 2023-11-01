from gui.Gui import Gui

# Beschikbare model types zijn: simple, rnn, en rnn_fma
# rnn_fma is onder constructie en werkt nog niet

gui = Gui(model_type="rnn")
gui.run()
