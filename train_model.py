import json
from pprint import pprint
import csv
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import read_data
import features
import itertools
# SemEval
data_folder_sem_eval = 'data/sem_eval/'


# Map to ANGER, JOY, SADNESS, SUPRISE
mapping = {
    'fairy_tales': {'2': 'ANGER', '3': 'NONE', '4': 'JOY', '5': 'NONE', '6': 'SADNESS', '7': 'SURPRISE'},
    'ISEAR': {'anger': 'ANGER', 'disgust': 'NONE', 'fear': 'NONE', 'joy': 'JOY', 'sadness': 'SADNESS', 'shame': 'NONE', 'guilt': 'NONE'},
    'sem_eval': {'anger': 'ANGER', 'disgust': 'ANGER', 'fear': 'NONE', 'joy': 'JOY', 'sadness': 'SADNESS', 'surprise': 'SURPRISE'},
    'facebook': {'ANGRY': 'ANGER', 'SAD': 'SADNESS', 'HAHA': 'JOY', 'WOW': 'SURPRISE', 'LOVE': 'JOY', 'LIKE': 'NONE', 'THANKFUL': 'NONE', 'NONE' : 'NONE'}
}


def read_facebook_data(pages):
    reaction_types = ['NONE', 'LIKE', 'LOVE',
                      'WOW', 'HAHA', 'SAD', 'ANGRY', 'THANKFUL']
    data = []
    for f in pages:
        with open('data/facebook_data/{}.json'.format(f)) as data_file:
            page = json.load(data_file)
            data.extend(page)
    csv_data = []
    categories = []
    text = defaultdict(str)
    result = defaultdict(list)
    for post in data:
        if post[0]['reactions'][0] > 0:
            csv_row = [post[0]['created_time'], post[0]['message']]
            for i, rt in enumerate(reaction_types):
                csv_row.append(post[0]['reactions'][i] / post[0]['reactions'][0])
            csv_row.append(reaction_types[post[0]['reactions'].index(
                max(post[0]['reactions'][2:]))])
            csv_data.append(csv_row)
            categories.append(
                reaction_types[post[0]['reactions'].index(max(post[0]['reactions'][2:]))])
            text[reaction_types[post[0]['reactions'].index(
                max(post[0]['reactions'][2:]))]] += post[0]['message']
            result[mapping['facebook'][reaction_types[post[0]['reactions'].index(max(post[0]['reactions'][2:]))]]].append((mapping['facebook'][reaction_types[
                post[0]['reactions'].index(max(post[0]['reactions'][2:]))]], post[0]['message'], max(post[0]['reactions'][2:]) / post[0]['reactions'][0]))

    least_frequent = min([len(result[k])for k in result])
    sorted_result = []
    for k in result:

        # print(result[k])
        sorted_result.extend(
            sorted(result[k], key=lambda x: int(x[2])))
            #could limit the list to least_frequent to balance classes
    # return [(y,x,s) for y,x,s in sorted_result if s > 0.15]
    
    # print("{} {} {} {} {}".format(pages[0], len(result['ANGER']),len(result['JOY']),len(result['SADNESS']),len(result['SURPRISE'])))
    return sorted_result

def classify_and_evaluate(train, test):
    X = [x for y, x, s in train if y != 'NONE']
    y = [y for y, x, s in train if y != 'NONE']

    pipeline = Pipeline([
        ('features', FeatureUnion([
            # ('lexicon_nrc', features.NRCLexicon()),
            # ('words', TfidfVectorizer(stop_words='english')),
            # ('token_ngrams', TfidfVectorizer(stop_words='english', ngram_range = (2, 5))),
            # ('char_ngrams', TfidfVectorizer(ngram_range = (2, 5), analyzer = 'char')),
            # ('negation', features.Negation() ),  
            ('embeddings', features.Embeddings() ),
            # ('punctuation', features.Punctuation() ), 
            # ('retrofit', features.Retrofitted() ),  
        ])),
        ('classifier', svm.SVC(kernel='linear', C=0.5, class_weight = 'balanced'))
    ])
    pipeline.fit(X, y)
    X_test = [x for y, x in test if y != 'NONE']
    y_test = [y for y, x in test if y != 'NONE']
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
   

pages = ['time', 'theguardian', 'Disney'] # best model
# pages = ['HuffPostWeirdNews', 'ESPN', 'CNN'] # model 2 fairy tales
# pages = ['time', 'theguardian', 'CookingLight'] # model 3 ISEAR
train = read_facebook_data(pages)


# train = read_data.read_newsgroup_train()
# test = read_data.read_newsgroup_test()


print("Test on SemEval train")
test = read_data.read_sem_eval_train()
classify_and_evaluate(train, test)
print()

# print("Test on SemEval test")
# test = read_data.read_sem_eval_test()
# classify_and_evaluate(train, test)
# print()

# print("Test on ISEAR")
# test = read_data.read_isear()
# classify_and_evaluate(train, test)
# print()

# print("Test on FairyTales")
# test = read_data.read_fairy_tales()
# classify_and_evaluate(train, test)
# print()


