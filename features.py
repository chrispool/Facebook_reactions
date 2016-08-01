from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from nltk.stem import WordNetLemmatizer


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
        
        
     
