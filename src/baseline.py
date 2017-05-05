import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from data import eval_ds
from model_utils import Model


class SimpleBaseline(Model):
    porter = nltk.PorterStemmer()

    def _predict(self, a, b):
        a = {SimpleBaseline.porter.stem(x) for x in a}
        b = {SimpleBaseline.porter.stem(x) for x in b}
        return len(a & b) / len(a | b)

    def preprocessLine(self, sentence):
        return nltk.word_tokenize(sentence)


model = SimpleBaseline("log", eval_ds, {})
model.evalScore()
