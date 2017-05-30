import os
from src.char_lstm.lstm import LSTM
from src.char_lstm.proj_path import project_path
from src.char_lstm.text import Text
from data import eval_ds

from nltk.corpus import webtext, abc, gutenberg

# file_name = "input.txt"
# file = Text.open_file(file_name)

# file_list = [webtext.raw("overheard.txt"), abc.raw("rural.txt"), gutenberg.raw("carroll-alice.txt")]
file = webtext.raw("overheard.txt")
task = Text(file)


class Hp:
    lstm_memory_size = 512
    out_vector_size = task.num_chars
    n_layers = 2
    temp = 0.7

    batch_size = 128
    seq_len = 64
    steps = 10000


model_info = ["log_lstm", eval_ds, {"log_overwrite:": True}]
model = LSTM(Hp.lstm_memory_size, Hp.n_layers, Hp.out_vector_size, Hp.temp, task, model_info)

model.train(Hp, project_path)

# model.restore_path = os.path.join(project_path.base, project_path.log_dir, "May_30__14:22", "train", "model.chpt")
model.restore_path = project_path.model_path + "-" + str(Hp.steps)
model.eval_score_tf()
