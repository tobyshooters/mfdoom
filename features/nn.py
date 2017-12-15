from collections import defaultdict, Counter
import sqlite3
import re
import math
import ast
import pprint
pp = pprint.PrettyPrinter()
import util
import features_util
import numpy as np
from scipy import stats
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

emolex = util.parseEmoLex("./data/emolex.txt")

# TODO:
# Number of distinct rhyming phonemes
# Number of rhyming syllable pairs
acceptable_verse_types = ["hook", "chorus", "verse", "bridge", "intro", "outro",\
        "prechorus", "postchorus", "prehook", "posthook", "interlude", "refrain", "drop"]

def extractFeatures(song, total_doc_count):
# Maybe incorporate parts of speech with NLTK
    title, artist, raw_lyrics, _, _ = song
    lyrics = ast.literal_eval(raw_lyrics)

    verse_types = defaultdict(int)
    affect_categories = defaultdict(int)
    word_count = defaultdict(int)
    number_count = defaultdict(int)
    not_word_count = defaultdict(int)
    excl_count = 0
    ques_count = 0
    line_count = 0
    number_stanzas = 1

    # Parts of Speech
    flat_lyrics = ["".join(re.findall(r"[A-Za-z0-9]+", word)) for line in lyrics \
            for word in line if "".join(re.findall(r"[A-Za-z0-9]+", word))]

    tagged = nltk.pos_tag(flat_lyrics, tagset="universal")
    pos_counts = Counter(tag for word, tag in tagged)
    del pos_counts["X"]
    del pos_counts["."]

    for line in lyrics:
        if not line:
            continue
        elif len(line) == 1 and line[0][1:] in acceptable_verse_types:
            number_stanzas += 1
            verse_types[line[0].lower()] +=1
        else:
            # Emo-lex
            line_count += 1
            for word in line:
                excl_count += word.count('!')
                ques_count += word.count('?')
                alphanum = "".join(re.findall(r"[A-Za-z0-9]+", word))
                if re.match(r"[0-9]", alphanum):
                    number_count[alphanum] += 1
                word_count[alphanum] += 1
                if alphanum not in emolex:
                    not_word_count[alphanum] += 1
                for affect in emolex[alphanum]:
                    affect_categories[affect] += 1

    # Singleton Features
    total = sum(word_count.values()) * 1.0
    distinct = len(word_count.keys()) * 1.0

    tf_idf = 0
    for term in word_count:
        if term in total_doc_count:
            tf = word_count[term] * 1.0 / total
            idf = math.log(3390.0 / total_doc_count[term])
            tf_idf += tf * idf

    features = {
            "word_count": total,
            "!_count": excl_count,
            "?_count": ques_count,
            "number_count": sum(number_count.values()),
            "not_words": sum(not_word_count.values()) / total,
            "distinct_words": distinct,
            "distinct_words_per_line": distinct / line_count,
            "vocab_salience": tf_idf,
            "richness": distinct / total,
            "stanzas": number_stanzas,
            "avg_stanzas": total / number_stanzas,
            "lines": line_count,
            "avg_line": total / line_count
            }

    # Too sparse!
    popular_words = {"pop_word " + word: 1 \
            for word in sorted(word_count, key=word_count.get, reverse=True)[:3]}

    # https://pudding.cool/2017/09/hip-hop-words/
    common_words = {
            "common_i": util.perc(word_count["i"], total),
            "common_we": util.perc(word_count["we"], total),
            "common_us": util.perc(word_count["us"], total),
            "common_love": util.perc(word_count["love"], total),
            "common_bitch": util.perc(word_count["bitch"], total),
            "common_fuck": util.perc(word_count["fuck"], total),
            "common_money": util.perc(word_count["money"], total),
            "common_rap": util.perc(word_count["rap"], total)
            }

    # Normalization
    util.normalize_vector(verse_types)
    util.normalize_vector(affect_categories)
    util.normalize_vector(pos_counts)

    # pp.pprint(util.merge_dicts(verse_types, affect_categories, pos_counts, features))
    return util.merge_dicts(
            verse_types, 
            affect_categories, 
            pos_counts,
            features)


def getFeatures(cached, database, limit):
    if not cached:
        print "Not cached, producing new features"
        features_util.createDataset(database, extractFeatures, limit)

    print "Getting raw features from cache"
    titles_train, X_train, Y_train, titles_test, X_test, Y_test = features_util.getCachedDataset(database)

    vec = DictVectorizer()
    vec.fit(X_train + X_test)
    X_all = vec.transform(X_train + X_test)
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)

    Y_total = Y_train + Y_test
    Y_train = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_train])
    Y_test = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_test])
    Y_all = np.array([stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_total])

    return titles_train, X_train, np.reshape(Y_train, (len(Y_train), 1)), titles_test, X_test, np.reshape(Y_test, (len(Y_test), 1)), X_all, Y_all
