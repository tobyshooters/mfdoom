import re
import sqlite3
from dateutil import parser
import billboard

# NOTE: using onyl peakPosition rather than average of positions
db = sqlite3.connect("../data/1990")
cursor = db.cursor()
# cursor.execute(''' 
#         CREATE TABLE songs(id INTEGER PRIMARY KEY, title TEXT unique, artist TEXT, last_date TEXT, peak INTEGER, weeks INTEGER)
#         ''')
# db.commit()

def cleanArtist(artist):
    alphanumeric = " ".join(re.findall(r"[a-z0-9!?\s-]+", artist.lower()))
    tokens = [el for el in alphanumeric.split(" ") if el]
    final = []
    for tk in tokens:
        if re.match(r'(feat|ft)', tk): 
            break
        else: 
            final.append(tk)
    return " ".join(final) if final else "!NO_ARTIST"

# Get all songs
# Charts: Rap and RnB + Hip-Hop
def getSongs(chart_name):
    final_date = parser.parse("1990-01-01") # end date
    current_date = "2008-03-08" # start date
    chart = billboard.ChartData(chart_name, current_date) 

    while chart.previousDate:
        print current_date

        for song in chart:
            artist = cleanArtist(song.artist)
            cursor.execute('''
            INSERT OR IGNORE INTO songs(title, artist, last_date, peak, weeks) VALUES(?, ?, ?, ?, ?)
            ''', (song.title, artist, current_date, song.peakPos, song.weeks))
            db.commit()

        prev = parser.parse(chart.previousDate)
        if prev < final_date: break

        current_date = chart.previousDate
        chart = billboard.ChartData(chart_name, current_date)

    print current_date
    print "Finished"

# getSongs('rap-song')
getSongs('r-b-hip-hop-songs')
