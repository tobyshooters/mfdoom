import util
from collections import defaultdict, Counter
import sqlite3
import re
import ast
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from scipy.spatial import distance as dist
import pprint
pp = pprint.PrettyPrinter()

emolex = util.parseEmoLex("./data/emolex.txt")
acceptable_verse_types = ["hook", "chorus", "verse", "bridge", "intro", "outro",\
        "prechorus", "postchorus", "prehook", "posthook", "interlude", "refrain", "drop"]

def extractFeatures(song):
# Maybe incorporate parts of speech with NLTK
    title, artist, raw_lyrics, _, _ = song
    lyrics = ast.literal_eval(raw_lyrics)

    verse_types = defaultdict(int)
    affect_categories = defaultdict(int)
    word_count = defaultdict(int)
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
                alphanum = "".join(re.findall(r"[A-Za-z0-9]+", word))
                word_count[alphanum] += 1
                for affect in emolex[alphanum]:
                    affect_categories[affect] += 1

    # Singleton Features
    total = sum(word_count.values()) * 1.0
    distinct = len(word_count.keys()) * 1.0

    features = {
            "word_count": total,
            "distinct_words": distinct,
            "distinct_words_per_line": distinct / line_count,
            "richness": distinct / total,
            "stanzas": number_stanzas,
            "avg_stanzas": total / number_stanzas,
            "lines": line_count,
            "avg_line": total / line_count
            }

    # Normalization
    util.normalize_vector(verse_types)
    util.normalize_vector(affect_categories)
    util.normalize_vector(pos_counts)
    return  util.merge_dicts(verse_types, affect_categories, pos_counts, features)

def createDataset(extract_fn, limit):
    db = sqlite3.connect("data/1990")
    c = db.cursor()
    songs = c.execute(''' SELECT title, artist, lyrics, peak, weeks 
            FROM songs WHERE lyrics is not NULL {}'''.format(limit)).fetchall()

    titles = []
    X = []
    Y = []
    for i, s in enumerate(songs):
        X.append(extract_fn(s))
        Y.append(util.calculateScore(s))
        titles.append(s[0])
        if i % 25 == 0: 
            print "Features done: ", i 

    with open("data/cluster", "w") as f:
        f.write(str(X) + "\n")
        f.write(str(Y) + "\n")
        f.write(str(titles) + "\n")

def getDataset():
    with open("data/cluster", "r") as f:
        X = ast.literal_eval(f.readline())
        Y = ast.literal_eval(f.readline())
        titles = ast.literal_eval(f.readline())
    return X, Y, titles

def lossCluster(centroid, values):
    total_loss = 0
    num = 0
    for val in values:
        total_loss += dist.euclidean(centroid, val.todense())
        num += 1
    print "Number:       ", num
    print "Total Loss:   ", total_loss
    print "Average Loss: ", total_loss * 1.0 / num

def cluster():
    # createDataset(extractFeatures, "")
    X, Y, titles = getDataset()
    # Sparsity
    vec = DictVectorizer()
    sparse_X = vec.fit_transform(X)
    # Scaling
    scaler = MaxAbsScaler()
    scaled_X = scaler.fit_transform(sparse_X)

    for i in range(3, 20):
        km = KMeans(n_clusters=i, max_iter=300, n_init=5)
        km.fit(scaled_X)

        print "========================================================="
        print "Number of Clusters", i, km.inertia_

        for cluster, cluster_val in enumerate(km.cluster_centers_):
            print "------------------------"
            print "Cluster:      ", cluster
            lossCluster(cluster_val, [scaled_X[n] for n, label in enumerate(km.labels_) if label == cluster])
            # num_elems = sum([1 for n, label in enumerate(km.labels_) if label == cluster])

if __name__ == '__main__':
    cluster()
