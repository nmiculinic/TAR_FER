"""
If you get this error after importing gensim:
"Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so"
try running:
conda install -f numpy
from the command line and it magically fixes it.
"""
import gensim
import logging
import numpy as np
import os
from dotenv import load_dotenv


logger = logging.getLogger(__file__)

base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.environ.get(
    "TAR_DATA_ENV",
    os.path.join(base, 'src', 'local.env')
))

w2v_path = os.path.join(base, os.environ['W2V_PATH'])
eval_data_in_path = os.path.join(base, os.environ["EVAL_IN"])
eval_data_out_path = os.path.join(base, os.environ["EVAL_OUT"])


class DataSet:
    def __init__(self, in_file, target_file, name):
        with open(target_file) as fn:
            self.target = np.array([float(val) for val in fn.readlines()])

        with open(in_file) as fn:
            self.input = [pair.strip().split("\t") for pair in fn.readlines()]

        self.name = name


eval_ds = DataSet(eval_data_in_path, eval_data_out_path, "Eval dataset")

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
except FileNotFoundError:
    logger.error("Cannot find w2v path %s", w2v_path, exc_info=False)
