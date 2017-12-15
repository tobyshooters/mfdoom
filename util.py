# GENERAL FUNCTIONALITIES
from collections import defaultdict
import ast
import re
import sqlite3
import pprint
pp = pprint.PrettyPrinter()
from multiprocessing import Pool
import numpy as np

def createDataset(name, extract_fn, limit):
    print "Features set: ", name

    db = sqlite3.connect("data/1990")
    c = db.cursor()
    train_songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
                            FROM songs WHERE lyrics is not NULL 
                            AND date(last_date) < date('2015-01-01') {}'''.format(limit)).fetchall()

    test_songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
                            FROM songs WHERE lyrics is not NULL
                            AND date(last_date) >= date('2015-01-01') {}'''.format(limit)).fetchall()


    songs = train_songs + test_songs

    print "Getting doc counts for words"
    total_doc_count = docCounts(songs)

    def _operate(s):
        title = s[0]
        features = extract_fn(s, total_doc_count)
        score = calculateScore(s)
        return (title, features, score)

    def _getTFS(raw_set):
        p = Pool(10)
        return p.map(_operate, raw_set)

    def getTitlesFeaturesScores(raw_set):
        titles = []
        X = []
        Y = []
        threads = []
        for i, s in enumerate(raw_set):
            X.append(extract_fn(s, total_doc_count))
            Y.append(calculateScore(s))
            titles.append(s[0])
            if i % 25 == 0: 
                print "Features done: ", i 

        return titles, X, Y

    titles_train, X_train, Y_train = getTitlesFeaturesScores(train_songs)
    titles_test, X_test, Y_test = getTitlesFeaturesScores(test_songs)

    print "Caching features..."
    cacheDataset(name, titles_train, X_train, Y_train, titles_test, X_test, Y_test)
    print "Done"

def calculateScore(song):
    # Normalizes best score and multiplies by number of weeks
    _, _, _, peak, weeks = song
    normalized_peak = ((101 - peak * 1.0)**2)/10000
    score = normalized_peak * weeks
    return score

def docCounts(songs):
    # Number of songs in which a word appears
    doc_count = defaultdict(int)
    for _, _, raw_lyrics, _, _ in songs:
        word_count = defaultdict(int)
        lyrics = ast.literal_eval(raw_lyrics)
        for line in lyrics:
            for word in line:
                alphanum = "".join(re.findall(r"[^A-Za-z0-9]+", word))
                word_count[alphanum] += 1
        for word in word_count:
            doc_count[word] += 1
    return doc_count

def cacheDataset(f, titles_train, X_train, Y_train, titles_test, X_test, Y_test):
    # Writes raw data to file
    with open(f, "w") as f:
        f.write(str(titles_train) + "\n")
        f.write(str(X_train) + "\n")
        f.write(str(Y_train) + "\n")
        f.write(str(titles_test) + "\n")
        f.write(str(X_test) + "\n")
        f.write(str(Y_test))

def getCachedDataset(f):
    # Reads raw data using ast from file
    with open(f, "r") as f:
        titles_train = ast.literal_eval(f.readline())
        X_train = ast.literal_eval(f.readline())
        Y_train = ast.literal_eval(f.readline())
        titles_test = ast.literal_eval(f.readline())
        X_test = ast.literal_eval(f.readline())
        Y_test = ast.literal_eval(f.readline())
    return titles_train, X_train, Y_train, titles_test, X_test, Y_test

def exampleSongs(extract_fn):
    db = sqlite3.connect("data/1990")
    c = db.cursor()
    train_songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
                            FROM songs WHERE lyrics is not NULL
                            AND title LIKE 'Gucci Gang' OR title LIKE 'm.A.A.d City' ''').fetchall()
    for song in train_songs:
        print song[0]
        pp.pprint(extract_fn(song, defaultdict(int)))

def parseEmoLex(path):
    # Generates dictionary from Emolex
    words = defaultdict(list)
    with open(path, 'r') as emolex:
        for line in emolex:
            word, category, flag = line.strip().split("\t")
            if int(flag):
                words[word].append(category)
    return words

def chunks(l, n):
    # Partitions list into chunks of length n
    for i in range(0, len(l), n):
        yield l[i:i + n]

def merge_dicts(*dict_args):
    # Combines dictionaries
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def normalize_vector(vector):
    # Normalizes sparse vectors
    total = sum(vector.values())
    for k, v in vector.items():
        vector[k] = v * 1.0 / total

def perc(count, total):
    return 100.0 * count / total
