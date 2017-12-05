import re
import sqlite3
from scipy import stats
import pprint

pp = pprint.PrettyPrinter()
db = sqlite3.connect("data/1990")
c = db.cursor()

def calculateScore(peak, weeks):
    normalized_peak = ((101 - peak * 1.0)**2)/10000
    score = normalized_peak * weeks
    return score

def baseline(title, artist):
    # Escape Error
    if artist.find("\"") != -1 or title.find("\"") != -1: return 0
    same_artist = c.execute(''' SELECT peak, weeks FROM songs
                                WHERE artist LIKE "{}" and title != "{}" '''.format(artist, title))
    songs = same_artist.fetchall()
    if songs:
        scores = [calculateScore(peak, weeks) for peak, weeks in songs]
        return sum(scores) / len(scores)
    else:
        return 0

def evaluate():
    songs = c.execute(''' SELECT title, artist, peak, weeks FROM songs''').fetchall()
    results = []
    for title, artist, peak, weeks in songs:
        actual = calculateScore(peak, weeks)
        base = baseline(title, artist)
        if base:
            results.append((actual, base))

    scores = [r[0] for r in results]
    predict = [r[1] for r in results]
    scores = [stats.percentileofscore(scores, a, 'rank') / 100.0 for a in scores]
    predict = [stats.percentileofscore(predict, a, 'rank') / 100.0 for a in predict]
    final = zip(scores, predict)

    tae = 0
    for a, b in final:
        tae += a - b if a > b else b - a

    mae = tae / len(final)
    print mae

evaluate()
