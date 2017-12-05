# GENERAL FUNCTIONALITIES
from collections import defaultdict
import ast
import re
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

def createDataset(name, extract_fn, limit):
    print "Features set: ", name

    db = sqlite3.connect("data/1990")
    c = db.cursor()
    songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
                            FROM songs WHERE lyrics is not NULL {}'''.format(limit)).fetchall()

    titles = []
    raw_scores = []
    raw_features = []

    print "Getting doc counts for words"
    total_doc_count = docCounts(songs)

    for i, s in enumerate(songs):
        raw_features.append(extract_fn(s, total_doc_count))
        raw_scores.append(calculateScore(s))
        titles.append(s[0])
        if i % 100 == 0: 
            print "Features done: ", i 

    print "Caching features..."
    cacheDataset(name, titles, raw_features, raw_scores)
    print "Done"
    return titles, raw_features, raw_scores

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

def cacheDataset(f, titles, features, scores):
    # Writes raw data to file
    with open(f, "w") as f:
        f.write(str(titles) + "\n")
        f.write(str(features) + "\n")
        f.write(str(scores))

def getCachedDataset(f):
    # Reads raw data using ast from file
    with open(f, "r") as f:
        titles = ast.literal_eval(f.readline())
        raw_features = ast.literal_eval(f.readline())
        scores = ast.literal_eval(f.readline())
    return titles, raw_features, scores

def visualizeScores(scores):
    # Provides histogram of scores
    hist, bins = np.histogram(scores, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

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
