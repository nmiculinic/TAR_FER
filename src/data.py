import gensim

"""
If you get this error after importing gensim:
"Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so"
try running:
conda install -f numpy
from the command line and it magically fixes it.
"""
import numpy as np
from utils import *

word2vec = gensim.models.KeyedVectors.load_word2vec_format(proj_path.w2v, binary=False, limit=10000)


class EvaluationData:
    def __init__(self):
        self.raw_input = open(proj_path.eval_data_in).read()
        # Dividing target with 5 to get it in the [0, 1] range
        self.target = np.array([float(val) for val in open(proj_path.eval_data_out).read().split()]) / 5

        self.proc_input = self.input_process()
        # We can store the entire evaluation data vectors in memory
        # ~250 sentences * 2* (10 words?) * 300 dimension * 4 bytes =~ 6mb
        self.input_vectors = self.dataset_to_vector()

    def input_process(self):
        """
        Just doing a lot of splitting
        :return: a list with elements [sentence1, sentence2]. Each of those contains a list of words. 
        Basically it returns a triple-nested list
        """
        # Just doing a lot of splitting
        inp = [pair.split("\t") for pair in self.raw_input.split("\n")][:-1]
        for i, pair in enumerate(inp):
            inp[i][0] = pair[0].split()
            inp[i][1] = pair[1].split()
        return inp

    def dataset_to_vector(self):
        """
        
        :return: Whole dataset converted to a word embedding form
        Ideally, it should be a numpy array of shape [n_pairs, 2, n_words, embedding_dimension] but not all sentences 
        are the same length so numpy arranges it somehow. :)
        """
        sv = self.sentence_to_vector
        return np.array([[sv(pair[0]), sv(pair[1])] for pair in self.proc_input])

    @staticmethod
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
