import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from data import eval_ds, sentence_to_vector
from model_utils import Model

class SimpleBaseline(Model):
    porter = nltk.PorterStemmer()

    def _predict(self, a, b):
        fn = np.sum
        sem1 = fn(a, axis=0)
        sem2 = fn(b, axis=0)
        return cosine(sem1, sem2)

    def preprocessLine(self, sentence):
        return sentence_to_vector(nltk.word_tokenize(sentence))

model = SimpleBaseline("log_cosine", eval_ds, {'log_overwrite':True})
model.evalScore()
