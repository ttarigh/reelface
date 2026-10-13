import sqlite3
import re
import os

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()
c.execute("SELECT id, creator_name, title, video_url, like_count FROM videos ORDER BY like_count DESC")
all_videos = c.fetchall()

print(f"Total videos in DB: {len(all_videos)}")

# 1. Flag all static photo posts (/p/)
photo_posts = []
for v in all_videos:
    if "/p/" in v[3]:
        photo_posts.append(v)

# 2. Flag known non-talking-head accounts (athletes, singers, actors without talking-head monologues, dance channels, AI accounts, memes)
bad_handles = {
    'cristiano', 'neymarjn.br', 'shawnmendes', 'jlo', 'dollyparton', 'sadiesink_',
    'alvarocarreras', 'realmadrid', 'candasifamabarnes', 'candasifama', 'prithika.chow...',
    'prithika.chowdhury', 'jakobowens', 'simonegiertz', 'mrbeast', 'chandlerhallow',
    'jennaortega', 'milliebobbybrown', 'georginagio', 'coryxkenshin', 'monkshen',
    'kick.core7', 'amineo_ai', 'longliveai', 'artikumarimodi_', 'smartypantsdo',
    'made.of.lines', 'creativetoons111', 'kunanets.art', 'artintoline', 'mysteryvault_0',
    'time_less51', 'benazelart', 'lauren.welds', 'thecolorgirlm...', 'thecolorgirl',
    'reesebros', 'darienj4', 'macklemore', 'senatorhawley', 'lifewith.tripp',
    'candasifamabarnes', 'tipsrahul_', 'doctor.tomeo', 'advsanafkhan', '_truthdecodedd'
}

rejected_by_handle = []
for v in all_videos:
    h = v[1].lstrip('@').lower()
    for b in bad_handles:
        if b in h:
            rejected_by_handle.append(v)
            break

all_flagged = set(p[0] for p in photo_posts) | set(r[0] for r in rejected_by_handle)
print(f"Flagged for removal: {len(all_flagged)}")
print(f"Photo posts: {len(photo_posts)}")
print(f"Non-talking-head creators: {len(rejected_by_handle)}")

# Delete them from DB
print("Deleting flagged non-talking-head rows...")
for vid in all_flagged:
    c.execute("DELETE FROM videos WHERE id = ?", (vid,))
conn.commit()

c.execute("SELECT COUNT(*) FROM videos")
remaining = c.fetchone()[0]
print(f"Clean remaining videos in DB: {remaining}")
needed = 500 - remaining
print(f"Need {needed} verified talking-head replacements to reach 500.")
conn.close()
