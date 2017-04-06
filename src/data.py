import gensim
import nltk
import logging

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
from copy import deepcopy

logger = logging.getLogger(__file__)

base = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.environ.get(
    "TAR_DATA_ENV",
    os.path.join(base, 'src', 'local.env')
))

w2v_path = os.path.join(base, os.environ['W2V_PATH'])
eval_data_in_path = os.path.join(base, os.environ["EVAL_IN"])
eval_data_out_path = os.path.join(base, os.environ["EVAL_OUT"])


class DataSet:
    def __init__(self, in_file, target_file, sentence_preprocess_fn=None):
        if sentence_preprocess_fn is not None:
            self.preprocessLine = sentence_preprocess_fn

        with open(target_file) as fn:
            # Dividing target with 5 to get it in the [0, 1] range
            self.target = np.array([float(val) for val in fn.readlines()])

        with open(in_file) as fn:
            inp = [pair.strip().split("\t") for pair in fn.readlines()]
            self.raw_input = deepcopy(inp)
            for i, pair in enumerate(inp):
                inp[i][0] = self.preprocessLine(pair[0])
                inp[i][1] = self.preprocessLine(pair[1])
            self.input = inp

    def preprocessLine(self, sentence):
        raise NotImplemented


eval_ds = DataSet(eval_data_in_path, eval_data_out_path, nltk.word_tokenize)

try:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False, limit=10000)

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

    w2w_ds = DataSet(eval_data_in_path, eval_data_out_path, lambda x: sentence_to_vector(nltk.word_tokenize(x)))

except FileNotFoundError:
    logger.error("Cannot find w2v path %s", w2v_path, exc_info=False)
