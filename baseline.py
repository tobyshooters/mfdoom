import re
import sqlite3
import pprint

pp = pprint.PrettyPrinter()
db = sqlite3.connect("data/hiphop")
c = db.cursor()

def calculateScore(peak, weeks):
    normalized_peak = ((101 - peak * 1.0)**2)/10000
    score = normalized_peak * weeks
    return score

def baseline(title, artist):
    # Escape Error
    if artist.find("\"") != -1 or title.find("\"") != -1: return 0
    same_artist = c.execute(''' SELECT peak, weeks FROM songs
                                WHERE artist = "{}" and title != "{}" '''.format(title, artist))
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

    pp.pprint(results)
    mae = sum([a - b for a, b in results]) / len(results)
    print mae

evaluate()
