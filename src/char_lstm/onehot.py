import os
import numpy as np
from src.char_lstm.proj_path import project_path


def flatten(x): return [x for x in x for x in x]


class OneHot:
    def __init__(self, file_list):
        self.file_list = file_list
        self.files_train, self.files_test = [], []
        for file in self.file_list:
            split_point = int(0.75 * len(file))
            self.files_train.append(file[:split_point])
            self.files_test.append(file[split_point:])

        self.chars = sorted(list(set(flatten(self.file_list))))
        self.num_chars = len(self.chars)
        self.char_to_ind = {c: ind for ind, c in enumerate(self.chars)}
        self.ind_to_char = {ind: c for c, ind in self.char_to_ind.items()}

        self.file_train_ind = [[self.char_to_ind[char] for char in file] for file in self.files_train]
        self.file_test_ind = [[self.char_to_ind[char] for char in file] for file in self.files_test]

    def next_batch(self, batch_size, seq_len=20, test=False):
        if test:
            source = self.file_test_ind
        else:
            source = self.file_train_ind
        file_ind = np.random.randint(0, len(source))
        file = source[file_ind]
        indices = np.random.randint(0, len(file) - seq_len, batch_size)
        x = np.array([file[ind:ind + seq_len] for ind in indices])
        return x

    def text_to_indices(self, text):
        x = np.expand_dims([self.char_to_ind[char] for char in text], axis=0)
        return x

    def sample_text(self, sess, next_char_tf, next_state_tf, x, initial_state, seed=None, length=20):
        if seed is None:
            next_char = np.array([[np.random.randint(0, self.num_chars)]])
        else:
            next_char = self.text_to_indices(seed)
        next_state = None
        sampled_sentence = self.ind_to_char[next_char[0, 0]]
        for i in range(length):
            feed_dict = {x: next_char}
            if next_state is not None:
                feed_dict[initial_state] = tuple(next_state)
            next_char, next_state = sess.run([next_char_tf, next_state_tf], feed_dict=feed_dict)
            char = self.ind_to_char[next_char[0, 0]]
            sampled_sentence += char
        return sampled_sentence, next_state

    @staticmethod
    def open_file(file_name):
        text_path = os.path.join(project_path.base, file_name)
        return open(text_path).read()
