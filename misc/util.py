# GENERAL FUNCTIONALITIES
from collections import defaultdict

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
