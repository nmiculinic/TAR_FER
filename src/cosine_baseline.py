from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from data import eval_ds, sentence_to_vector
from model_utils import Model


class SimpleBaseline(Model):
    porter = nltk.PorterStemmer()

    def _predict(self, a, b):
        fn = np.sum
        sem1 = fn(a, axis=0).reshape(1, -1)
        sem2 = fn(b, axis=0).reshape(1, -1)
        return cosine_similarity(sem1, sem2)[0, 0]

    def preprocessLine(self, sentence):
        return sentence_to_vector(nltk.word_tokenize(sentence))


model = SimpleBaseline("log_cosine", eval_ds, {'log_overwrite': True})
model.evalScore()
