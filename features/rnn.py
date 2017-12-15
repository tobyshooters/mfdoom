from collections import defaultdict, Counter
import sqlite3
import re
import ast
import pprint
pp = pprint.PrettyPrinter()
import util
import features_util
import numpy as np
import math
from scipy import stats
from scipy.sparse import csc_matrix, vstack
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer

emolex = util.parseEmoLex("./data/emolex.txt")

def extractFeatures(song, total_doc_count):
    # For each song, return a list of features corresponding to each line of the song
    # Note that the length of the list varies and therefore it will need to be padded
    title, artist, raw_lyrics, _, _ = song
    lyrics = ast.literal_eval(raw_lyrics)

    song_temporal_features = []
    for line in lyrics:
        affect_categories = defaultdict(int)
        word_count = defaultdict(int)
        number_count = defaultdict(int)
        not_word_count = defaultdict(int)
        excl_count = 0
        ques_count = 0

        # PART-OF-SPEECH FEATURES
        flat_lyrics = ["".join(re.findall(r"[A-Za-z0-9]+", word)) for word in line \
                if "".join(re.findall(r"[A-Za-z0-9]+", word))]

        tagged = nltk.pos_tag(flat_lyrics, tagset="universal")
        pos_counts = Counter(tag for word, tag in tagged)
        del pos_counts["X"]
        del pos_counts["."]

        # EMOLEX FEATURES
        for word in line:
            excl_count += word.count('!')
            ques_count += word.count('?')
            alphanum = "".join(re.findall(r"[^A-Za-z]+", word))
            if re.match(r"[0-9]", alphanum):
                number_count[alphanum] += 1
            word_count[alphanum] += 1
            if alphanum not in emolex:
                not_word_count[alphanum] += 1
            for affect in emolex[alphanum]:
                affect_categories[affect] += 1


        total = sum(word_count.values())
        distinct = len(word_count.keys())

        if total == 0:
            continue

        tf_idf = 0
        for term in word_count:
            if term in total_doc_count:
                tf = word_count[term] * 1.0 / total
                idf = math.log(3390.0 / total_doc_count[term])
                tf_idf += tf * idf

        # GENERIC LINE FEATURES
        line_features = {
                "word_count": total,
                "!_count": excl_count,
                "?_count": ques_count,
                "number_count": sum(number_count.values()),
                "not_words": sum(not_word_count.values()) / total,
                "distinct_words": distinct,
                "vocab_salience": tf_idf,
                "richness": distinct * 1.0 / total,
                }

        util.normalize_vector(pos_counts)
        util.normalize_vector(affect_categories)

        features = util.merge_dicts(affect_categories, pos_counts, line_features)
        song_temporal_features.append(features)

    return song_temporal_features

def getFeatures(cached, limit):
    if not cached:
        features_util.createDataset("data/rnn_features", extractFeatures, limit)
    titles_train, X_train, Y_train, titles_test, X_test, Y_test = features_util.getCachedDataset("data/rnn_features")

    vec = DictVectorizer()
    total_features = X_train + X_test
    flat_features = [feat for seq in total_features for feat in seq]
    vec.fit(flat_features)

    def makeSparse(raw_features):
        features = []
        for seq in raw_features:
            limited = seq[:200]
            pad_size = 200 - len(limited)
            sparse = vec.transform(limited)
            padding = csc_matrix((pad_size, sparse[0].shape[1]), dtype=np.float64)
            padded_seq = vstack((sparse, padding))
            features.append(padded_seq.todense())
        return features

    X_all = makeSparse(X_train + X_test)
    X_train = makeSparse(X_train)
    X_test = makeSparse(X_test)

    Y_total = Y_train + Y_test
    Y_train = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_train])
    Y_test = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_test])
    Y_all = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_total])

    return titles_train, np.array(X_train), np.reshape(Y_train, (len(Y_train), 1)), titles_test, np.array(X_test), np.reshape(Y_test, (len(Y_test), 1)), np.array(X_all), np.array(Y_all)
