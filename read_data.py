from xml.dom.minidom import parse, parseString
from collections import Counter, defaultdict
import os

# Map to ANGER, JOY, SADNESS, SUPRISE
mapping = {
    'fairy_tales': {'2': 'ANGER', '3': 'NONE', '4': 'JOY', '5': 'NONE', '6': 'SADNESS', '7': 'SURPRISE'},
    'ISEAR': {'anger': 'ANGER', 'disgust': 'NONE', 'fear': 'NONE', 'joy': 'JOY', 'sadness': 'SADNESS', 'shame': 'NONE', 'guilt': 'NONE'},
    'sem_eval': {'anger': 'ANGER', 'disgust': 'ANGER', 'fear': 'NONE', 'joy': 'JOY', 'sadness': 'SADNESS', 'surprise': 'SURPRISE'},
    'facebook': {'ANGRY': 'ANGER', 'SAD': 'SADNESS', 'HAHA': 'JOY', 'WOW': 'SURPRISE', 'LOVE': 'JOY', 'LIKE': 'NONE', 'NONE': 'NONE'}
}


def read_fairy_tales():
    data_folder = 'data/Fairy_tales/agreed/'
    result_fairy_tales = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".agree"):
            with open(data_folder + '/' + filename, 'r') as f:
                for line in f.readlines():
                    row = line.strip().split("@")
                    result_fairy_tales.append(
                        (mapping['fairy_tales'][row[1]], row[2]))
    return result_fairy_tales


def read_isear():
    data_folder_ISEAR = 'data/ISEAR/'
    result_ISEAR = []
    with open(data_folder_ISEAR + '/data.csv', 'r') as f:
        for line in f.readlines():
            row = line.strip().split("---")
            result_ISEAR.append((mapping['ISEAR'][row[1]], row[2]))
    return result_ISEAR


def read_sem_eval_train():
    train_gold_doc = parse(
        'data/sem_eval/AffectiveText.trial/affectivetext_trial.xml')
    train_sentences = 'data/sem_eval/AffectiveText.trial/affectivetext_trial.emotions.gold'
    emotions = ['anger', 'disgust', 'fear',
                'joy', 'sadness', 'surprise', 'sentence']

    result_sem_eval = defaultdict(list)
    with open(train_sentences) as f2:
        for line in f2.readlines():
            values = line.strip().split()
            result_sem_eval[values[0]] = [
                emotions[values[1:].index(max(values[1:]))]]

    sentences = train_gold_doc.getElementsByTagName("instance")
    for sentence in sentences:
        sid = sentence.getAttribute("id")
        result_sem_eval[sid].append(sentence.firstChild.nodeValue)

    result_sem_eval = [(mapping['sem_eval'][emotion], sentence)
                       for emotion, sentence in result_sem_eval.values()]
    return result_sem_eval


def read_sem_eval_test():
    '''Reading sem_eval test'''
    test_gold_doc = parse(
        'data/sem_eval/AffectiveText.test/affectivetext_test.xml')
    test_sentences = 'data/sem_eval/AffectiveText.test/affectivetext_test.emotions.gold'
    emotions = ['anger', 'disgust', 'fear',
                'joy', 'sadness', 'surprise', 'sentence']

    result_sem_eval = defaultdict(list)
    with open(test_sentences) as f2:
        for line in f2.readlines():
            values = line.strip().split()
            result_sem_eval[values[0]] = [
                emotions[values[1:].index(max(values[1:]))]]

    sentences = test_gold_doc.getElementsByTagName("instance")
    for sentence in sentences:
        sid = sentence.getAttribute("id")
        result_sem_eval[sid].append(sentence.firstChild.nodeValue)

    result_sem_eval = [(mapping['sem_eval'][emotion], sentence)
                       for emotion, sentence in result_sem_eval.values()]
    return result_sem_eval
