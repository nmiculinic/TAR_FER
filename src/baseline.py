import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from data import EvaluationData

sol = []

porter = nltk.PorterStemmer()


def find_similarity(a, b):
    a = {porter.stem(x) for x in a}
    b = {porter.stem(x) for x in b}
    return len(a & b) / len(a | b)


data = EvaluationData()

sol = []
for (xa, xb), y in zip(data.input, data.target):
    sol.append((xa, xb, y, find_similarity(xa, xb)))

df = pd.DataFrame(sol, columns=["A", "B", "target", "pred"])
pearson_corr = df['target'].corr(df['pred'])
mse = np.mean((df["target"] - df["pred"])**2)
print("correlation coeficient is", pearson_corr)
print("MSE is ", mse)
plt.plot(df["target"], df["pred"], 'o')
plt.show()
