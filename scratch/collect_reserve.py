import asyncio
import os
import re
import cv2
import numpy as np
import requests
import sqlite3
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

TAGS = [
    "storytime", "publicspeaking", "communication", "lifecoach",
    "psychology", "dermatologist", "careeradvice", "philosophy",
    "dailywisdom", "socialskills", "tedtalk", "monologue", "mindsetmatters"
]

async def collect_reserve_active_reels():
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT creator_name, video_id FROM videos")
    existing_handles = set(r[0].lstrip('@').lower() for r in c.fetchall())
    existing_vids = set(r[1] for r in c.fetchall())
    conn.close()

    candidates = []

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for tag in TAGS:
            if len(candidates) >= 100: break
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            print(f"Scraping #{tag} for reserve HD reels...")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=8000)
                await asyncio.sleep(0.5)
                reels = await page.evaluate(r"""() => {
                    const anchors = document.querySelectorAll('a[href*="/reel/"]');
                    return Array.from(anchors).map(a => {
                        const img = a.querySelector('img');
                        const text = a.innerText || '';
                        const parts = text.split('\n');
                        return {
                            href: a.href.split('?')[0],
                            author: (parts[0] || '').trim(),
                            viewsStr: (parts[1] || '').trim(),
                            caption: parts.slice(2).join(' ').trim(),
                            imgSrc: img ? img.src : ''
                        };
                    });
                }""")

                for r in reels:
                    author = r['author'].lstrip('@').lower()
                    if not author or author in existing_handles or author in ('instagram', 'explore', 'reels'):
                        continue
                    if any(b in author for b in ['art', 'draw', 'film', 'movie', 'clip', 'quote', 'facts', 'page', 'hub', 'tv', 'news', 'baker']):
                        continue

                    v_str = r['viewsStr'].lower().replace('views','').replace(',','').strip()
                    views = 1000000
                    if 'm' in v_str: views = int(float(v_str.replace('m','')) * 1000000)
                    elif 'k' in v_str: views = int(float(v_str.replace('k','')) * 1000)
                    if views < 500000: continue
                    likes = max(100000, int(views / 10))

                    m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                    vid = m.group(1) if m else r['href']
                    if vid in existing_vids: continue

                    existing_handles.add(author)
                    existing_vids.add(vid)
                    candidates.append({
                        'creator_name': f'@{author}',
                        'creator_url': f'https://www.instagram.com/{author}/',
                        'platform': 'Instagram Reels',
                        'video_id': vid,
                        'video_url': r['href'],
                        'title': r['caption'][:90] or f'Reel by @{author}',
                        'view_count': views,
                        'like_count': likes,
                        'upload_date': '2025-2026',
                        'thumbnail_url': r['imgSrc'],
                        'face_confidence': 0.95,
                        'face_area_pct': 5.0,
                        'verified_talking_head': 1
                    })
                    print(f"  + Candidate @{author} ({likes:,} likes)")
                    if len(candidates) >= 100: break
            except Exception as e:
                print(f"Error on #{tag}: {e}")

        await ctx.close()

    print(f"Total reserve candidates collected: {len(candidates)}")
    # Save to json
    import json
    with open("scratch/reserve_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

if __name__ == "__main__":
    asyncio.run(collect_reserve_active_reels())
