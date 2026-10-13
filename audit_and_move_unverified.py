import asyncio
import os
import re
import json
import shutil
import sqlite3
from playwright.async_api import async_playwright

CREATORS_PATH = "assets/creators.json"
PHOTOS_DIR = "talking_head_photos_100"
OLD_UNVERIFIED_DIR = "old_unverified"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
UNVERIFIED_MANIFEST = "assets/unverified_manifest.json"

os.makedirs(OLD_UNVERIFIED_DIR, exist_ok=True)

async def worker(wid, queue, verified, unverified, lock):
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        page = await ctx.new_page()

        while not queue.empty():
            try:
                creator = await queue.get()
            except asyncio.QueueEmpty:
                break

            rank = creator['rank']
            handle = creator['handle']
            url = creator.get('reelUrl', '')
            photo = creator.get('photo', '')
            vid = url.rstrip('/').split('/')[-1] if '/reel/' in url else ''

            likes = -1
            if vid:
                try:
                    await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=4000)
                    await asyncio.sleep(0.4)
                    text = await page.evaluate("() => document.body.innerText")
                    for line in text.split('\n'):
                        m = re.search(r'([0-9,]+)\s+likes', line, re.I)
                        if m:
                            likes = int(m.group(1).replace(',', ''))
                            break
                except Exception:
                    pass

            item = {
                'rank': rank,
                'handle': handle,
                'reelUrl': url,
                'video_id': vid,
                'photo': photo,
                'likes': likes
            }

            photo_path = os.path.join(PHOTOS_DIR, photo)
            unverified_path = os.path.join(OLD_UNVERIFIED_DIR, photo)

            if likes >= 50000:
                async with lock:
                    verified[rank] = item
                print(f"  [W{wid}] ✓ #{rank} @{handle:18} -> {likes:,} likes [KEEP]", flush=True)
            else:
                async with lock:
                    unverified[rank] = item
                # Move unverified photo out of active directory into old_unverified
                if os.path.exists(photo_path):
                    shutil.move(photo_path, unverified_path)
                print(f"  [W{wid}] ✗ #{rank} @{handle:18} -> {likes:,} likes [MOVED TO old_unverified]", flush=True)

            queue.task_done()

        await page.close()
        await ctx.close()

async def run_audit():
    with open(CREATORS_PATH) as f:
        creators = json.load(f)

    print(f"=== Starting Strict 50K Likes Audit Across {len(creators)} Creators ===")

    queue = asyncio.Queue()
    for c in creators:
        queue.put_nowait(c)

    verified = {}
    unverified = {}
    lock = asyncio.Lock()

    num_workers = 8
    workers = [asyncio.create_task(worker(i, queue, verified, unverified, lock)) for i in range(num_workers)]

    async def monitor():
        while not queue.empty():
            await asyncio.sleep(5.0)
            async with lock:
                with open(VERIFIED_MANIFEST, 'w') as f:
                    json.dump(verified, f, indent=2)
                with open(UNVERIFIED_MANIFEST, 'w') as f:
                    json.dump(unverified, f, indent=2)
            print(f"--> Audit Progress: {len(verified)} verified (>=50K), {len(unverified)} moved to old_unverified...", flush=True)

    monitor_task = asyncio.create_task(monitor())
    await asyncio.gather(*workers)
    monitor_task.cancel()

    with open(VERIFIED_MANIFEST, 'w') as f:
        json.dump(verified, f, indent=2)
    with open(UNVERIFIED_MANIFEST, 'w') as f:
        json.dump(unverified, f, indent=2)

    print(f"\n=======================================================")
    print(f"AUDIT COMPLETE!")
    print(f"Verified (>=50,000 likes): {len(verified)}")
    print(f"Moved to old_unverified (<50,000 likes / unverified): {len(unverified)}")
    print(f"=======================================================")

if __name__ == "__main__":
    asyncio.run(run_audit())
