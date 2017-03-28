import os
import utils
import pandas as pd
import nltk
import numpy as np
sol = []

porter = nltk.PorterStemmer()
def preprocessLine(sentence):
    return [porter.stem(t) for t  in nltk.word_tokenize(sentence)]

def find_similarity(a, b):
    a = {x for x in a}
    b = {x for x in b}
    return len(a & b)/len(a | b)

with open(os.path.join(utils.repo_root, 'data', 'train-en-en.in')) as train_in, \
    open(os.path.join(utils.repo_root, 'data', 'train-en-en.out')) as train_out:
    for line, target in zip(train_in.readlines(), train_out.readlines()):
        a, b = list(map(lambda x: preprocessLine(x.strip()), line.split('\t')))
        sol.append((a,b, float(target), find_similarity(a,b)))

df = pd.DataFrame(sol, columns=["A","B", "target", "pred"])
# sns.lmplot(x="target", y="pred", data=df)
print("correlation coeficient is", df['target'].corr(df['pred']))