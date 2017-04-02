import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from data import *

data = EvaluationData()

cos_similarities = []
for i, (sent1, sent2) in enumerate(data.input_vectors):
    # we need a way to aggregate word embeddings within a sentence. np.mean? np.sum? a deep neural network?
    fn = np.sum
    sem1 = fn(sent1, axis=0)
    sem2 = fn(sent2, axis=0)
    cos_similarities.append(cosine(sem1, sem2))
cos_similarities = np.array(cos_similarities)

pearson_corr = np.corrcoef([data.target, cos_similarities])
print("Correlation coefficitent:", pearson_corr[0, 1])
plt.plot(data.target, cos_similarities, "o")
plt.show()
