from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from data import eval_ds, sentence_to_vector
from model_utils import Model


class SimpleBaseline(Model):
    porter = nltk.PorterStemmer()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def _predict(self, a, b):
        fn = np.sum
        sem1 = fn(a, axis=0).reshape(1, -1)
        sem2 = fn(b, axis=0).reshape(1, -1)
        return cosine_similarity(sem1, sem2)[0, 0]

    def preprocess_line(self, sentence):
        tokens = [token for token in nltk.word_tokenize(sentence) if token not in self.stopwords]
        return sentence_to_vector(tokens)


model = SimpleBaseline("log_cosine", eval_ds, {'log_overwrite': True})
model.eval_score()
