import sqlite3
import re

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()
c.execute("SELECT id, creator_name, title, video_url, like_count FROM videos ORDER BY like_count DESC")
videos = c.fetchall()

print(f"Total videos in DB: {len(videos)}")

# Strict list of words that indicate non-human, theme page, meme edit, movie, or clip
REJECT_KEYWORDS = [
    "marvel", "hbo", "espn", "fifa", "wasted", "digitaltrends", "apple", "3d print",
    "spiderman", "netflix", "primevideo", "movie", "film", "oppenheimer", "trailer",
    "tamasha", "i, robot", "bbcearth", "bbc", "advice", "heritagecdjr", "taylornation",
    "football", "realmadrid", "ferrari", "robot", "animation", "cartoon", "requiem mod",
    "uncanny", "horror", "baker", "pastry", "recipe", "food", "contractor", "welder"
]

# Specifically check if the creator handle or title matches non-person
rejected = []
approved = []

for v in videos:
    vid, name, title, url, likes = v
    handle = name.lstrip("@").lower()
    title_lower = title.lower()
    
    # Check if known brand / movie / clip aggregator
    is_rejected = False
    for kw in [
        'marvelstudios', 'hbomax', 'espn', 'fifaworldcup', 'wasted', 'digitaltrends',
        'appledsign', 'unnecessaryinventions', 'spiderman', 'netflixisajoke', 'primevideoin',
        'eatmovies', 'filmaze', 'goodmovie_greatmovie', 'mdsc.films', 'notsoserious143',
        'politibrawlstreetcast', 'heritagecdjr_parkville', 'taylornation', 'realmadrid',
        'bbcearth', 'bbc', 'advice', 'wtfakul', 'thephysics_explorer', 'chomuki_rai',
        'yurii.thebuilder', 'patelmedical85', 'r6katanaa', 'with.kuttippa', 'sunny_.vloger',
        'travelerjeet42', 'khixarrhere', 'joshua_spisak', 'onsidianmen', 'taylorcutfilms'
    ]:
        if kw in handle:
            is_rejected = True
            break
            
    if not is_rejected:
        # Check title for third-person movie clips
        if any(term in title_lower for term in ['trailer for', 'i, robot', 'oppenheimer directed by', 'movie🍿', 'welcome home, vision', 'gabriel macht']):
            is_rejected = True

    if is_rejected:
        rejected.append(v)
    else:
        approved.append(v)

print(f"Approved genuine talking heads: {len(approved)}")
print(f"Rejected non-people / movie / clip edits: {len(rejected)}")
print("\nSample rejected:")
for r in rejected[:15]:
    print(f"  {r[0]}: {r[1]} | {r[2][:50]} | {r[3]}")
