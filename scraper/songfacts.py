import json
import urllib2
from bs4 import BeautifulSoup
import pprint
pp = pprint.PrettyPrinter()

def getSongList():
    songs = []
    for index in range(1, 18):
        url = "http://www.songfacts.com/released-2017-{}.php".format(index)
        page = urllib2.urlopen(url)
        html = BeautifulSoup(page, "html.parser")
        song_list = html.find("ul", attrs={"class": "songullist-orange"})
        for s in song_list:
            songs.append(s.get_text().lower())
    return songs

pp.pprint(getSongList())
