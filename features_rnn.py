from collections import defaultdict, Counter
import sqlite3
import re
import ast
import pprint
pp = pprint.PrettyPrinter()
import util
import numpy as np
from scipy import stats
from scipy.sparse import csc_matrix, vstack
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer

emolex = util.parseEmoLex("./data/emolex.txt")

def extractFeatures(song, total_word_count):
    # For each song, return a list of features corresponding to each line of the song
    # Note that the length of the list varies and therefore it will need to be padded
    title, artist, raw_lyrics, _, _ = song
    lyrics = ast.literal_eval(raw_lyrics)

    song_temporal_features = []
    for line in lyrics:
        word_count = defaultdict(int)
        affect_categories = defaultdict(int)

        # EMOLEX FEATURES
        for word in line:
            alphanum = "".join(re.findall(r"[^A-Za-z]+", word))
            word_count[alphanum] += 1
            for affect in emolex[alphanum]:
                affect_categories[affect] += 1
        util.normalize_vector(affect_categories)

        # PART-OF-SPEECH FEATURES
        flat_lyrics = ["".join(re.findall(r"[^A-Za-z0-9]+", word)) for line in lyrics for word in line]
        tagged = nltk.pos_tag(flat_lyrics, tagset="universal")
        pos_counts = Counter(tag for word, tag in tagged)
        del pos_counts["X"]
        del pos_counts["."]
        util.normalize_vector(pos_counts)

        # GENERIC LINE FEATURES
        line_features = {
                "word_count": sum(word_count.values()),
                "distinct_words": len(word_count.keys())
                }

        features = util.merge_dicts(affect_categories, pos_counts, line_features)
        song_temporal_features.append(features)

    return song_temporal_features

def getFeatures(cached, limit):
    if cached:
        print "Getting raw features from cache"
        titles, raw_features, raw_scores = util.getCachedDataset("data/recurrent_features")
    else:
        print "Not cached, producing new features"
        titles, raw_features, raw_scores = util.createDataset("data/recurrent_features", extractFeatures, limit)

    vec = DictVectorizer()
    all_features = [feat for seq in raw_features for feat in seq]
    vec.fit(all_features)

    features = []
    for seq in raw_features:
        limited = seq[:200]
        pad_size = 200 - len(limited)
        sparse = vec.transform(limited)
        padding = csc_matrix((pad_size, sparse[0].shape[1]), dtype=np.float64)
        padded_seq = vstack((sparse, padding))
        features.append(padded_seq.todense())

    perc_scores = [stats.percentileofscore(raw_scores, a, 'rank') / 100.0 for a in raw_scores]
    util.visualizeScores(perc_scores)
    scores = np.array(perc_scores)

    return titles, np.array(features), np.reshape(scores, (len(scores), 1))
