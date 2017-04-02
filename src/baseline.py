import os
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from utils import *

sol = []

porter = nltk.PorterStemmer()


def preprocessLine(sentence):
    return [porter.stem(t) for t in nltk.word_tokenize(sentence)]


def find_similarity(a, b):
    a = {x for x in a}
    b = {x for x in b}
    return len(a & b) / len(a | b)


with open(proj_path.eval_data_in) as train_in, \
        open(proj_path.eval_data_out) as train_out:
    for line, target in zip(train_in.readlines(), train_out.readlines()):
        a, b = list(map(lambda x: preprocessLine(x.strip()), line.split('\t')))
        sol.append((a, b, float(target)/5, find_similarity(a, b)))

df = pd.DataFrame(sol, columns=["A", "B", "target", "pred"])
# sns.lmplot(x="target", y="pred", data=df)
pearson_corr = df['target'].corr(df['pred'])
mse = np.mean((df["target"] - df["pred"])**2)
print("correlation coeficient is", )
print("MSE is ", mse)
plt.plot(df["target"], df["pred"], 'o')
plt.show()
