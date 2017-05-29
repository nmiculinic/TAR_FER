import nltk
from data import eval_ds
from model_utils import Model


class SimpleBaseline(Model):
    porter = nltk.PorterStemmer()

    def _predict(self, a, b):
        a = {SimpleBaseline.porter.stem(x) for x in a}
        b = {SimpleBaseline.porter.stem(x) for x in b}
        return len(a & b) / len(a | b)

    def preprocess_line(self, sentence):
        return nltk.word_tokenize(sentence)

model = SimpleBaseline("log", eval_ds, {})
model.eval_score()
