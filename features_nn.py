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
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

emolex = util.parseEmoLex("./data/emolex.txt")

acceptable_verse_types = ["hook", "chorus", "verse", "bridge", "intro", "outro",\
        "prechorus", "postchorus", "prehook", "posthook", "interlude", "refrain", "drop"]

# TODO:
# Number of distinct rhyming phonemes
# Number of rhyming syllable pairs

def extractFeatures(song):
    # Maybe incorporate parts of speech with NLTK
    title, artist, raw_lyrics, _, _ = song
    lyrics = ast.literal_eval(raw_lyrics)
    tokenizer = RegexpTokenizer(r'\w+')
    token_lyrics = [tokenizer.tokenize(line) for line in lyrics]

    verse_types = defaultdict(int)
    affect_categories = defaultdict(int)
    word_count = defaultdict(int)
    line_count = 0
    number_stanzas = -3 # Tokenizing Bug

    # Parts of Speech
    flat_lyrics = [word.lower() for line in lyrics for word in line]
    tagged = nltk.pos_tag(flat_lyrics, tagset="universal")
    pos_counts = Counter(tag for word, tag in tagged)
    del pos_counts["X"]
    del pos_counts["."]

    for line in token_lyrics:
        if not line:
            # Number of verses
            number_stanzas += 1
        elif len(line) == 1 and line[0].lower() in acceptable_verse_types:
            verse_types[line[0].lower()] +=1
        else:
            # Emo-lex
            line_count += 1
            for word in line:
                word = re.sub(r"[^A-Za-z]+", '', word)
                word_count[word] += 1
                for affect in emolex[word]:
                    affect_categories[affect] += 1

    # Singleton Features
    features = {
            "word_count": sum(word_count.values()),
            "distinct_words": len(word_count.keys()),
            "stanzas": number_stanzas,
            "avg_stanzas": sum(word_count.values()) / number_stanzas,
            "lines": line_count,
            "avg_line": sum(word_count.values()) / line_count
            }

    # Too sparse!
    popular_words = {"pop_word " + word: 1 \
            for word in sorted(word_count, key=word_count.get, reverse=True)[:3]}

    # Normalization
    util.normalize_vector(verse_types)
    util.normalize_vector(affect_categories)
    util.normalize_vector(pos_counts)

    # pp.pprint(util.merge_dicts(verse_types, affect_categories, pos_counts, features))
    return util.merge_dicts(verse_types, affect_categories, pos_counts, features)

def getFeatures(cached, limit):
    if cached:
        raw_features, raw_scores = util.getCachedDataset("data/features2")
    else:
        raw_features, raw_scores = util.createDataset("data/features2", extractFeatures, limit)

    vec = DictVectorizer()
    features = vec.fit_transform(raw_features)

    perc_scores = [stats.percentileofscore(raw_scores, a, 'rank') / 100.0 for a in raw_scores]
    scores = np.array(perc_scores)

    return features, np.reshape(scores, (len(scores), 1))
