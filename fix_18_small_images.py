import asyncio
import os
import re
import sqlite3
import cv2
import numpy as np
import requests
from playwright.async_api import async_playwright
from fast_stream_screenshot_generator import sample_video, clean_handle

small_ranks = [6, 18, 19, 21, 64, 90, 97, 102, 110, 129, 133, 148, 184, 189, 207, 276, 348, 436]

async def fix_small():
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT id, creator_name, video_id, video_url, like_count FROM videos ORDER BY like_count DESC")
    rows = c.fetchall()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for rank in small_ranks:
            r = rows[rank - 1]
            vid_id, creator_name, video_id, video_url, likes = r
            h = clean_handle(creator_name)
            out_p = f"talking_head_photos/{rank:03d}_{h}.jpg"
            print(f"\nProcessing Rank {rank:03d}: @{h} ({video_id})...")

            # Try embed
            await page.goto(f"https://www.instagram.com/reel/{video_id}/embed/", wait_until='domcontentloaded')
            await asyncio.sleep(0.4)
            v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
            
            saved = False
            if v_src:
                f = sample_video(v_src)
                if f is not None and min(f.shape[:2]) >= 320:
                    cv2.imwrite(out_p, f)
                    print(f"  ✓ SAVED DIRECT VIDEO: {out_p} {f.shape}")
                    saved = True

            if not saved:
                # Try high-res poster img from embed
                img_src = await page.evaluate(r"""() => {
                    const imgs = Array.from(document.querySelectorAll('img'));
                    // Filter out small avatars
                    const poster = imgs.find(img => img.naturalWidth >= 320 || img.width >= 320 || img.src.includes('scontent'));
                    return poster ? poster.src : null;
                }""")
                if img_src and img_src.startswith('http'):
                    try:
                        resp = requests.get(img_src, timeout=5)
                        if resp.status_code == 200:
                            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None and min(img.shape[:2]) >= 320:
                                cv2.imwrite(out_p, img)
                                print(f"  ✓ SAVED HQ POSTER: {out_p} {img.shape}")
                                saved = True
                    except Exception:
                        pass

            if not saved:
                print(f"  ✗ Failed to get HD frame for Rank {rank:03d}, will substitute with verified HD video")

        await browser.close()
    conn.close()

asyncio.run(fix_small())
