# GENERAL FUNCTIONALITIES
from collections import defaultdict
import ast
import matplotlib.pyplot as plt
import numpy as np

def calculateScore(song):
    # Normalizes best score and multiplies by number of weeks
    _, _, _, peak, weeks = song
    normalized_peak = ((101 - peak * 1.0)**2)/10000
    score = normalized_peak * weeks
    return score

def cacheDataset(f, features, scores):
    # Writes raw data to file
    with open(f, "w") as f:
        f.write(str(features) + "\n")
        f.write(str(scores))

def getCachedDataset(f):
    # Reads raw data using ast from file
    with open(f, "r") as f:
        raw_features = ast.literal_eval(f.readline())
        scores = ast.literal_eval(f.readline())
    return raw_features, scores

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
