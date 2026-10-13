import sqlite3
import re
import cv2
import numpy as np
import requests

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()

# 1. Identify non-people rows to delete
c.execute("SELECT id, creator_name, title, video_url FROM videos")
rows = c.fetchall()

bad_handles = {
    'marvelstudios', 'hbomax', 'espn', 'fifaworldcup', 'wasted', 'digitaltrends',
    'appledsign', 'unnecessaryinventions', 'spiderman', 'netflixisajoke', 'primevideoin',
    'eatmovies', 'filmaze', 'goodmovie_greatmovie', 'mdsc.films', 'notsoserious143',
    'politibrawlstreetcast', 'heritagecdjr_parkville', 'taylornation', 'realmadrid',
    'bbcearth', 'bbc', 'advice', 'wtfakul', 'thephysics_explorer', 'chomuki_rai',
    'yurii.thebuilder', 'patelmedical85', 'r6katanaa', 'with.kuttippa', 'sunny_.vloger',
    'travelerjeet42', 'khixarrhere', 'joshua_spisak', 'onsidianmen', 'taylorcutfilms'
}

to_delete_ids = []
for r in rows:
    cid, name, title, url = r
    handle = name.lstrip('@').lower()
    title_lower = title.lower()
    if handle in bad_handles or any(term in title_lower for term in ['trailer for', 'i, robot', 'oppenheimer directed by', 'movie🍿', 'welcome home, vision', 'gabriel macht']):
        to_delete_ids.append(cid)

print(f"Deleting {len(to_delete_ids)} non-people entries...")
for did in to_delete_ids:
    c.execute("DELETE FROM videos WHERE id = ?", (did,))
conn.commit()

c.execute("SELECT COUNT(*) FROM videos")
remaining = c.fetchone()[0]
print(f"Remaining valid videos: {remaining}")
needed = 500 - remaining
print(f"Need {needed} replacement human creators to reach 500.")

conn.close()
