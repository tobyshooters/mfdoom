import sys
sys.path.append('../')
import config
import re
import sqlite3
import requests
from util import chunks
from bs4 import BeautifulSoup
import pprint
pp = pprint.PrettyPrinter()

base_url = 'https://api.genius.com'
headers = {'Authorization': 'Bearer ' + config.access_token}

db = sqlite3.connect("../data/1990")
c = db.cursor()

def searchGenius(term):
    search_url = base_url + "/search"
    params = {'q': term}
    response = requests.get(search_url, params=params, headers=headers)
    return response.json()['response']['hits']

def searchApiPath():
    # Use search term to find songs
    db = sqlite3.connect("../data/final")
    c = db.cursor()
    songs = c.execute(''' SELECT title, artist FROM songs WHERE api_path is NULL ''').fetchall()
    # Re-run until no results for throttling
    if not songs:
        print "Done with api_path!"
    for i, (title, artist) in enumerate(songs):
        print title, artist
        hits = searchGenius(title)
        if hits:
            c.execute(''' UPDATE songs SET api_path=:path WHERE title=:title ''', 
                    {"path": hits[0]['result']['api_path'], "title": title})
        else:
            print "FAILED: ", title, artist
        if i % 100 == 0:
            db.commit()
    db.commit()

# Hard-coded where search fails
# c.execute(''' UPDATE songs SET api_path=:path WHERE title=:title ''', 
#         {"path": "/songs/2867962", "title": "PPAP (Pen-Pineapple-Apple-Pen)"})
# c.execute(''' UPDATE songs SET api_path=:path WHERE title=:title ''', 
#        {"path": "/songs/58343", "title": "HYFR (Hell Ya F*****g Right)"})
# c.execute(''' UPDATE songs SET api_path=:path WHERE title=:title ''', 
#         {"path": "/artists/139436" , "title": "Cash Me Outside (#CashMeOutside)"})

# /songs/2942139
def lyrics_from_api(song_api_path):
    response = requests.get(base_url + song_api_path, headers=headers).json()
    path = response["response"]["song"]["path"]
    page = requests.get("http://genius.com" + path)
    html = BeautifulSoup(page.text, "html.parser")
    [h.extract() for h in html('script')]
    lyrics = html.find("div", class_="lyrics").get_text() 
    return lyrics

acceptable_verse_types = ["hook", "chorus", "verse", "bridge", "intro", "outro",\
        "prechorus", "postchorus", "prehook", "posthook", "interlude", "refrain", "drop"]

def process_lyrics(lyrics):
    final = []
    for line in lyrics.split("\n"):
        if line:
            # Generic Processing
            line = re.findall(r'[a-z0-9!?]+', line.lower().replace("\'", ""))
            if line[0] in acceptable_verse_types:
                final.append(["*" + line[0]])
            else:
                final.append(line)
    return final
    #return [line for line in lyrics.split("\n") if (line[0] != "[" if line else True)]

# Test Lyric Scraping
l = lyrics_from_api('/songs/2942139')
c = process_lyrics(l)
pp.pprint(c)

def getLyrics():
    songs = c.execute(''' SELECT title, api_path FROM songs WHERE lyrics is NULL and api_path is NOT NULL''').fetchall()
    # Re-run until no results for throttling
    if not songs:
        print "Done with lyrics!"

    for i, (title, api_path) in enumerate(songs):
        print title
        raw = lyrics_from_api(api_path)
        lyrics = str(process_lyrics(raw))
        c.execute(''' UPDATE songs SET lyrics=? WHERE title=? ''', (lyrics, title))
        if i % 25 == 0:
            db.commit()
    db.commit()

from threading import Thread

def _getLyrics(songs, results, thread):
    lyrics = []
    for i, (title, api_path) in enumerate(songs):
        print "Thread: ", i, "Title: ", title
        lyrics.append((str(process_lyrics(lyrics_from_api(api_path))), title))
    results.append(lyrics)

def threadLyrics():
    all_songs = c.execute(''' SELECT title, api_path FROM songs WHERE lyrics is NULL and api_path is NOT NULL''').fetchall()
    # Re-run until no results for throttling
    if not all_songs:
        print "Done with lyrics!"

    for i, chunk in enumerate(chunks(all_songs, 100)):
        print "Iteration: ", i

        results = []
        threads = []
        for j, songs in enumerate(chunks(chunk, 10)):
            t = Thread(target=_getLyrics, args=(songs, results, j))
            print "Starting thread: ", j
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        flat_results = [r for result in results for r in result]
        c.executemany(''' UPDATE songs SET lyrics=? WHERE title=? ''', flat_results)
        db.commit()

# searchApiPath()
# threadLyrics()
