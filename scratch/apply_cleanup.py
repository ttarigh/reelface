import sqlite3
import re

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()

# 1. Delete non-people
bad_handles = {
    'marvelstudios', 'hbomax', 'espn', 'fifaworldcup', 'wasted', 'digitaltrends',
    'appledsign', 'unnecessaryinventions', 'spiderman', 'netflixisajoke', 'primevideoin',
    'eatmovies', 'filmaze', 'goodmovie_greatmovie', 'mdsc.films', 'notsoserious143',
    'politibrawlstreetcast', 'heritagecdjr_parkville', 'taylornation', 'realmadrid',
    'bbcearth', 'bbc', 'advice', 'wtfakul', 'thephysics_explorer', 'chomuki_rai',
    'yurii.thebuilder', 'patelmedical85', 'r6katanaa', 'with.kuttippa', 'sunny_.vloger',
    'travelerjeet42', 'khixarrhere', 'joshua_spisak', 'onsidianmen', 'taylorcutfilms'
}

c.execute("SELECT id, creator_name, title FROM videos")
rows = c.fetchall()

to_delete = []
for cid, name, title in rows:
    handle = name.lstrip('@').lower()
    title_lower = title.lower()
    if handle in bad_handles or any(t in title_lower for t in ['trailer for', 'i, robot', 'oppenheimer directed by', 'movie🍿', 'welcome home, vision', 'gabriel macht']):
        to_delete.append(cid)

print(f"Deleting {len(to_delete)} non-human/movie entries...")
for did in to_delete:
    c.execute("DELETE FROM videos WHERE id = ?", (did,))
conn.commit()

# Check how many needed
c.execute("SELECT COUNT(*) FROM videos")
count = c.fetchone()[0]
print(f"Videos remaining: {count}")
needed = 500 - count
print(f"Need {needed} replacements.")

conn.close()
