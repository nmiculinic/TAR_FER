from data import eval_ds, sentence_to_vector
from model_utils import Model


class W2V(Model):
    def __init__(self, model_info):
        super().__init__(*model_info)

    def preprocess_line(self, sentence):
        tokens = [token for token in nltk.word_tokenize(sentence) if token not in self.stopwords]
        return sentence_to_vector(tokens)
