from collections import defaultdict, Counter
import sqlite3
import re
import ast
import pprint
pp = pprint.PrettyPrinter()
import util
import numpy as np
from scipy import stats
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer

def extractFeatures(song):
    # For each song, return a list of features corresponding to each line of the song
    # Note that the length of the list varies and therefore it will need to be padded

    title, artist, raw_lyrics, _, _ = s
    lyrics = ast.literal_eval(raw_lyrics)
    tokenizer = RegexpTokenizer(r'\w+')
    token_lyrics = [tokenizer.tokenize(line) for line in lyrics]

    song_temporal_features = []
    for line in token_lyrics:
        word_count = defaultdict(int)
        affect_categories = defaultdict(int)

        # EMOLEX FEATURES
        for word in line:
            word = re.sub(r"[^A-Za-z]+", '', word)
            word_count[word] += 1
            for affect in emolex[word]:
                affect_categories[affect] += 1
        util.normalize_vector(affect_categories)

        # PART-OF-SPEECH FEATURES
        flat_lyrics = [word.lower() for word in line]
        tagged = nltk.pos_tag(flat_lyrics, tagset="universal")
        pos_counts = Counter(tag for word, tag in tagged)
        del pos_counts["X"]
        del pos_counts["."]
        util.normalize_vector(pos_counts)

        # GENERIC LINE FEATURES
        line_features = {
                "word_count": sum(word_count.values()),
                "distinct_words": len(word_counts.keys())
                }

        features = util.merge_dicts(affect_categories, pos_counts, line_features)
        song_temporal_features.append(features)

    return song_temporal_features

def createDataset(limit):
    db = sqlite3.connect("data/final")
    c = db.cursor()
    songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
                            FROM songs WHERE lyrics is not NULL {}'''.format(limit)).fetchall()

    raw_scores = []
    raw_features = []

    for i, s in enumerate(songs):
        raw_features.append(extractFeatures(s))
        raw_scores.append(util.calculateScore(s))
        #sequence.pad_sequence(X_train, maxlen)

    print "Caching features..."
    util.cacheDataset("data/recurrent_features", raw_features, raw_scores)
    print "Done"

    return raw_features, raw_scores

def getFeatures(cached, limit):
    if cached:
        raw_features, raw_scores = util.getCachedDataset("data/recurrent_features")
    else:
        raw_features, raw_scores = createDataset(limit)

    vec = DictVectorizer()
    features = vec.fit_transform(raw_features)

    perc_scores = [stats.percentileofscore(raw_scores, a, 'rank') / 100.0 for a in raw_scores]
    scores = np.array(perc_scores)

    return features, np.reshape(scores, (len(scores), 1))
