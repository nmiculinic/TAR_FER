import gensim
import nltk

"""
If you get this error after importing gensim:
"Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so"
try running:
conda install -f numpy
from the command line and it magically fixes it.
"""
import numpy as np
import os
from dotenv import load_dotenv


base = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.environ.get(
    "TAR_DATA_ENV",
    os.path.join(base, 'src', 'local.env')
))

w2v_path = os.path.join(base, os.environ['W2V_PATH'])
eval_data_in_path = os.path.join(base, os.environ["EVAL_IN"])
eval_data_out_path = os.path.join(base, os.environ["EVAL_OUT"])

try:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False, limit=10000)
except FileNotFoundError:
    pass


def sentence_to_vector(sentence):
    """

    :param sentence: a list of words
    :return:
    """
    embedding = []
    for word in sentence:
        try:
            embedding.append(word2vec[word.lower()])
        except KeyError:
            pass
    return np.array(embedding)


class EvaluationData:
    def __init__(self):
        with open(eval_data_out_path) as fn:
            # Dividing target with 5 to get it in the [0, 1] range
            self.target = np.array([float(val) for val in fn.readlines()]) / 5

        self.input = self.input_process(eval_data_in_path)
        # We can store the entire evaluation data vectors in memory
        # ~250 sentences * 2* (10 words?) * 300 dimension * 4 bytes =~ 6mb
        self.input_vectors = self.dataset_to_vector()

    def input_process(self, fname):
        """
        Just doing a lot of splitting
        :return: a list with elements [sentence1, sentence2]. Each of those contains a list of words.
        Basically it returns a triple-nested list
        """
        with open(fname, 'r') as fn:
            inp = [pair.strip().split("\t") for pair in fn.readlines()]
            for i, pair in enumerate(inp):
                inp[i][0] = nltk.word_tokenize(pair[0])
                inp[i][1] = nltk.word_tokenize(pair[1])
            return inp

    def dataset_to_vector(self):
        """

        :return: Whole dataset converted to a word embedding form
        Ideally, it should be a numpy array of shape [n_pairs, 2, n_words, embedding_dimension] but not all sentences
        are the same length so numpy arranges it somehow. :)
        """
        sv = sentence_to_vector
        return np.array([[sv(pair[0]), sv(pair[1])] for pair in self.input])
