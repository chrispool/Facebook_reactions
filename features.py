from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import gensim, os

model = gensim.models.Word2Vec.load('word2vec_models/facebookmodel-2.mod')
# model_google  = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'word2vec_models/GoogleNews-vectors-negative300.bin'), binary=True)
# model_retrofitted = {}
# with open('retrofitting/facebook_retrofitted.txt', 'r') as f:
#     for line in f.readlines():
#         cols = line.strip().split()
#         vector = []    
#         model_retrofitted[cols[0]] = [float(c) for c in cols[1:]]


class NRCLexicon(TransformerMixin):
    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lexicon = defaultdict(list)
        with open('data/NRC-Emotion-Lexicon/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt') as file:
            for line in file.readlines():
                cols = line.strip().split('\t')
                if len(cols) == 3:
                    self.lexicon[cols[0]].append(int(cols[2]))

    def transform(self, X, **transform_params):
        return [self.calculate_score(i) for i in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def calculate_score(self, sentence):
        fields = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness','surprise', 'trust']
        vector = [0] * 10

        for word in sentence.split():
            lemmatized = self.wordnet_lemmatizer.lemmatize(word.lower())
            if lemmatized in self.lexicon:
                # print(lemmatized, self.lexicon[lemmatized])
                vector = list(map(sum, zip(vector, self.lexicon[lemmatized])))
        return vector

class Negation(TransformerMixin):
    def transform(self, X, **transform_params):
        return [self.calculate_score(i) for i in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def calculate_score(self, sentence):
        negations = ['n \'t', 'not', 'never', 'no one', 'nobody', 'none', 'no']
        for n in negations:
            if n in sentence.strip().lower():
                return [1]
        return [0]


class Punctuation(TransformerMixin):
    def transform(self, X, **transform_params):
        return [self.calculate_punctuation_vector(i) for i in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def calculate_punctuation_vector(self, sentence):
        punctuation = ['!', '?', '.']
        # print(sentence, [sentence.count(p) for p in punctuation])
        return [sentence.count(p) for p in punctuation]


class Embeddings(TransformerMixin):      
    def transform(self, X, **transform_params):
        return [self.avg_vector(i) for i in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def avg_vector(self, sentence):
        l_vector = len(model['dog'])
        list_of_embeddings = [model[word] for word in sentence if word in model]
        averages = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]   
        if len(averages) != l_vector:
            averages = [0] * l_vector
        return averages


class Retrofitted(TransformerMixin):      
    def transform(self, X, **transform_params):
        return [self.avg_vector(i) for i in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def avg_vector(self, sentence):
        list_of_embeddings = [model_retrofitted[word.lower()] for word in sentence if word.lower() in model_retrofitted]
        averages = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]   
        l_vector = len(model_retrofitted['dog'])
        if len(averages) != l_vector:
            averages = [0] * l_vector
        return averages  

