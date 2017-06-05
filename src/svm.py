import nltk
import numpy as np
import re
import os
import seaborn as sns
import pandas as pd
from nltk.corpus.reader import WordNetError
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import brown, wordnet
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt


wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english"))

tagger = nltk.tag.pos_tag
frequency_list = FreqDist(i.lower() for i in brown.words())
all_words_count = 0
for i in frequency_list:
    all_words_count += frequency_list[i]


def get_words(sentence):
    return [i.strip('., ') for i in sentence.split(' ')]


with open('word_to_vec', 'r') as f:
    embeddings = {}
    for line in f.readlines():
        args = get_words(line.strip("\n\t "))
        embeddings[args[0]] = [float(i) for i in args[1:]]


def get_ngram(word, n):
    ngrams = []
    word_len = len(word)
    for i in range(word_len - n + 1):
        ngrams.append(word[i: i + n])
    return ngrams


def get_lists_intersection(s1, s2):
    s1_s2 = []
    for i in s1:
        if i in s2:
            s1_s2.append(i)
    return s1_s2


def overlap(sentence1_ngrams, sentence2_ngrams):
    s1_len = len(sentence1_ngrams)
    s2_len = len(sentence2_ngrams)
    if s1_len == 0 and s2_len == 0:
        return 0
    s1_s2_len = max(1, len(get_lists_intersection(sentence2_ngrams, sentence1_ngrams)))
    return 2 / (s1_len / s1_s2_len + s2_len / s1_s2_len)


def get_ngram_feature(sentence1, sentence2, n):
    sentence1_ngrams = []
    sentence2_ngrams = []

    for word in sentence1:
        sentence1_ngrams.extend(get_ngram(word, n))

    for word in sentence2:
        sentence2_ngrams.extend(get_ngram(word, n))

    return overlap(sentence1_ngrams, sentence2_ngrams)


def is_subset(s1, s2):
    for i in s1:
        if i not in s2:
            return False
    return True


def get_numbers_feature(sentence1, sentence2):
    s1_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence1))]
    s2_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence2))]
    s1_s2_numbers = []
    for i in s1_numbers:
        if i in s2_numbers:
            s1_s2_numbers.append(i)

    s1ands2 = max(len(s1_numbers) + len(s2_numbers), 1)
    return [np.log(1 + s1ands2), 2 * len(s1_s2_numbers) / s1ands2,
            is_subset(s1_numbers, s2_numbers) or is_subset(s2_numbers, s1_numbers)]


def get_shallow_features(sentence):
    counter = 0
    for word in sentence:
        if len(word) > 1 and (re.match("[A-Z].*]", word) or re.match("\.[A-Z]+]", word)):
            counter += 1
    return counter


def get_word_embedding(inf_content, word):
    if inf_content:
        return np.multiply(information_content(word), embeddings.get(word, np.zeros(300)))
    else:
        return embeddings.get(word, np.zeros(300))


def sum_embeddings(words, inf_content):
    vec = get_word_embedding(inf_content, words[0])
    for word in words[1:]:
        vec = np.add(vec, get_word_embedding(inf_content, word))
    return vec


def word_embeddings_feature(sentence1, sentence2):
    return cosine_similarity(unpack(sum_embeddings(sentence1, False)),
                             unpack(sum_embeddings(sentence2, False)))[0][0]


def information_content(word):
    return np.log(all_words_count / max(1, frequency_list[word]))


def unpack(param):
    return param.reshape(1, -1)


def weighted_word_embeddings_feature(sentence1, sentence2):
    return cosine_similarity(unpack(sum_embeddings(sentence1, True)),
                             unpack(sum_embeddings(sentence2, True)))[0][0]


def weighted_word_coverage(s1, s2):
    s1_s2 = get_lists_intersection(s1, s2)
    return np.sum([information_content(i) for i in s1_s2]) / np.sum([information_content(i) for i in s2])


def harmonic_mean(s1, s2):
    if s1 == 0 or s2 == 0:
        return 0
    return s1*s2/(s1+s2)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('A') or treebank_tag.startswith('JJ'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_synset(word):
    try:
        return wordnet.synset(word + "." + get_wordnet_pos(tagger([word])[0][1]) + ".01")
    except (WordNetError, IndexError):
        return 0


def wordnet_score(word, s2):
    if word in s2:
        return 1
    else:
        similarities = []
        for w in s2:
            try:
                value = get_synset(word).path_similarity(get_synset(w))
                if value is None:
                    value = 0
                similarities.append(value)
            except AttributeError:
                similarities.append(0)
        return np.max(similarities)


def wordnet_overlap(s1, s2):
    suma = 0
    for w in s1:
        suma += wordnet_score(w, s2)
    return suma / len(s2)


def feature_vector(a, b):
    fvec= []
    # Ngram overlap

    fvec.append(get_ngram_feature(a, b, 1))
    fvec.append(get_ngram_feature(a, b, 2))
    fvec.append(get_ngram_feature(a, b, 3))

    # WordNet-aug. overlap -
    fvec.append(harmonic_mean(wordnet_overlap(a, b), wordnet_overlap(b, a)))

    # Weighted word overlap -
    fvec.append(harmonic_mean(weighted_word_coverage(a, b),
                                      weighted_word_coverage(b, a)))
    # sentence num_of_words differences -
    fvec.append(abs(len(a) - len(b)))

    # summed word embeddings - lagano
    fvec.append(word_embeddings_feature(a, b))
    fvec.append(weighted_word_embeddings_feature(a, b))

    # Shallow NERC - lagano
    fvec.append(get_shallow_features(a))
    fvec.append(get_shallow_features(b))

    # Numbers overlap - returns list of 3 features
    fvec.extend(get_numbers_feature(a, b))
    return fvec


def pearson(y_true, y_predicted):
    return pearsonr(y_true, y_predicted)[0]


def nested_cv(name, scorer=None):
    train_x = []
    train_y = []
    with open('../data/train-en-en.in', 'r') as f:
        for line in f.readlines():
            train_x.append([get_words(i.strip('\n\t ')) for i in line.split('\t')])

    with open('../data/train-en-en.out', 'r') as f:
        for line in f.readlines():
            train_y.append(float(line.strip(' ')))

    NUM_TRIALS = 30
    new_train = []
    for i in train_x:
        new_train.append(np.array(feature_vector(i[0], i[1]), dtype=np.float64))
    scaler = StandardScaler()
    parameters = {'kernel': ['linear', 'rbf'], 'C': [2 ** i for i in range(-7, 7)],
                  'gamma': [10 ** i for i in range(-5, 3)]}

    new_train = scaler.fit_transform(new_train)
    # non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        print(i)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        clf = GridSearchCV(SVR(), parameters, n_jobs=-1, cv=inner_cv, scoring=scorer)
        # clf.fit(new_train, train_y)
        # non_nested_scores[i] = clf.best_score_

        nested_score = cross_val_score(clf, X=new_train, y=train_y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    # score_difference = non_nested_scores - nested_scores
    # print("Average difference of {0:6f} with std. dev. of {1:6f}."
    #       .format(score_difference.mean(), score_difference.std()))

    # print("Average non-nested: ", np.mean(non_nested_scores))
    print("Average nested: ", np.mean(nested_scores))
    print("Std nested: ", np.std(nested_scores))
    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    # plt.subplot(211)
    # non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([nested_line],
               [name],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Nested Cross Validation " + name + " score",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    # plt.subplot(212)
    # difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    # plt.legend([difference_plot],
    #            ["Non-Nested CV - Nested CV " + name + " score"],
    #            bbox_to_anchor=(0, 1, .8, 0))
    # plt.ylabel("score difference", fontsize="14")
    plt.show()


class SVMModel:
    porter = nltk.PorterStemmer()
    model = None
    scaler = None

    def train(self, train_x, train_y):
        new_train = []
        for i in train_x:
            new_train.append(np.array(feature_vector(i[0], i[1]), dtype=np.float64))
        self.scaler = StandardScaler()
        parameters = {'kernel': ['linear', 'rbf'], 'C': [2 ** i for i in range(-7, 7)],
                      'gamma': [10 ** i for i in range(-5, 3)]}
        svc = GridSearchCV(SVR(), parameters, n_jobs=-1, cv=KFold(n_splits=5, shuffle=True, random_state=42))

        svc.fit(self.scaler.fit_transform(new_train), train_y)
        self.model = svc.best_estimator_
        print(svc.best_estimator_.get_params())
        print("#"*50)

    def predict(self, x):
        predictions = []
        for i in x:
            predictions.append(self.predict_one(i)[0])
        return predictions

    def predict_one(self, x):
        fvec = feature_vector(x[0], x[1])
        return self.model.predict(self.scaler.transform(np.array(fvec).reshape(1, -1)))

    def eval_score(self, y_predicted, y_true, filename, correlation_fun):
        r = correlation_fun(y_predicted, y_true)[0]
        df = pd.DataFrame.from_dict({
            "model": y_predicted,
            "target": y_true
        })

        g = sns.jointplot(x="target", y="model", data=df, kind="reg", color="r", size=7)
        g.savefig(os.path.join(filename))
        return r


def custom_cv():
    X = []
    Y = []
    with open('../data/train-en-en.in', 'r') as f:
        for line in f.readlines():
            X.append([get_words(i.strip('\n\t ')) for i in line.split('\t')])

    with open('../data/train-en-en.out', 'r') as f:
        for line in f.readlines():
            Y.append(float(line.strip(' ')))

    # nested_cv(X, Y)
    train_scores = []
    test_scores = []
    train_test_differences = []
    for i in range(30):
        print(i)
        model = SVMModel()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        model.train(x_train, y_train)
        y_predicted = model.predict(x_test)
        # model.eval_score(y_predicted, y_test, "test_correlation_pearson.png", pearsonr)
        # test_score = pearsonr(y_predicted, y_test)[0]
        test_score = r2_score(y_test, y_predicted)
        test_scores.append(test_score)

        # train evaluation
        y_predicted = model.predict(x_train)
        # model.eval_score(y_predicted, y_train, "train_correlation_pearson.png", pearsonr)
        # train_score = pearsonr(y_predicted, y_train)[0]
        train_score = r2_score(y_train, y_predicted)
        train_scores.append(train_score)
        train_test_differences.append(abs(train_score - test_score))

    # first nested cv
    # Average difference of 0.005461 with std. dev. of 0.002443.

    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(train_scores, color='r')
    nested_line, = plt.plot(test_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(30), train_test_differences)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
               bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")
    plt.show()
    print("Average train: ", np.mean(train_scores))
    print("Average test: ", np.mean(test_scores))

print("Pearson")
nested_cv("Pearson", make_scorer(pearson))

print("R2")
nested_cv("R2")
