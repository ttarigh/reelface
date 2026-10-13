import asyncio
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
import os
from playwright.async_api import async_playwright

ctx_ssl = ssl._create_unverified_context()
detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar.onnx', '', (320, 320), score_threshold=0.5)

OUTPUT_DIR = 'talking_head_photos_100'
VERIFIED_MANIFEST = 'assets/verified_50k_manifest.json'
CREATORS_PATH = 'assets/creators.json'

TARGETS = [
    ('brookeschofield', 'storytime'),
    ('tanamongeau', 'storytime'),
    ('taraswrld', 'storytime_rant'),
    ('annaxsitar', 'lifestyle_storytime'),
    ('emiliekiser', 'lifestyle_storytime'),
    ('bomanizer', 'comedy_monologue'),
    ('katieflorence', 'comedy_monologue'),
    ('morgannn_wilson', 'storytime'),
    ('clarkepeoples', 'storytime_nyc'),
    ('hannahstocking', 'comedy_acting'),
    ('macaroniandmadi', 'comedy_couple'),
    ('chloeshih', 'tech_career_chat'),
    ('victoriabrowne', 'mental_health_chat')
]

def loosened_eval(fr):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) == 0: return False, 0
    f = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
    fx, fy, fw, fh = [int(v) for v in f[:4]]
    area = (fw * fh) / (w * h)
    conf = float(f[-1])
    if area < 0.035 or area > 0.45: return False, 0
    if conf < 0.70: return False, 0
    return True, conf * 100

async def run():
    with open(VERIFIED_MANIFEST) as f: manifest = json.load(f)
    existing_handles = {v['handle'].lower() for v in manifest.values()}

    with open('assets/rejected_blacklist.json') as f: bl = json.load(f)
    bl_handles = {h.lower() for h in bl.get('handles', [])}

    active = [t for t in TARGETS if t[0].lower() not in existing_handles and t[0].lower() not in bl_handles]
    print(f"Targeting {len(active)} top talking head creators...")

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for handle, category in active:
            print(f"\n[Scraping] @{handle}...")
            try:
                await page.goto(f"https://www.instagram.com/{handle}/reels/", wait_until='networkidle', timeout=12000)
                content = await page.content()
                reels = list(dict.fromkeys(re.findall(r'/reel/([a-zA-Z0-9_-]+)', content)))
                print(f"  Found {len(reels)} reel IDs for @{handle}")
                
                for vid in reels[:6]:
                    url = f"https://r.jina.ai/https://www.instagram.com/reel/{vid}/embed/"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    try:
                        with urllib.request.urlopen(req, timeout=6.0, context=ctx_ssl) as u:
                            md = u.read().decode('utf-8')
                        m_likes = re.search(r'\[([0-9,]+)\s+likes\]', md)
                        if not m_likes: m_likes = re.search(r'([0-9,]+)\s+likes', md)
                        likes = int(m_likes.group(1).replace(',', '')) if m_likes else 0
                        if likes < 50000: continue

                        # Block music/songs/movie clips
                        if any(w in md.lower()[:400] for w in ['song', 'singing', 'movie clip', 'cinema', 'trailer', 'stand up']):
                            continue

                        imgs = re.findall(r'!\[[^\]]*\]\((https://scontent[^\)]+)\)', md)
                        best_fr, best_sc = None, -1.0
                        for img_url in imgs:
                            if 's100x100' in img_url or 's150x150' in img_url: continue
                            try:
                                r_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                                with urllib.request.urlopen(r_img, timeout=6.0, context=ctx_ssl) as iu:
                                    arr = np.asarray(bytearray(iu.read()), dtype=np.uint8)
                                    fr = cv2.imdecode(arr, -1)
                                ok, sc = loosened_eval(fr)
                                if ok and sc > best_sc:
                                    best_sc = sc
                                    best_fr = fr.copy()
                            except: pass

                        if best_fr is not None:
                            with open(VERIFIED_MANIFEST) as mf: manifest = json.load(mf)
                            existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
                            next_rank = max(existing_ranks + [0]) + 1
                            rank_str = f"{next_rank:04d}"
                            out_name = f"{rank_str}_{handle}.jpg"
                            out_path = os.path.join(OUTPUT_DIR, out_name)
                            cv2.imwrite(out_path, best_fr)

                            manifest[rank_str] = {
                                'rank': rank_str,
                                'handle': handle,
                                'video_id': vid,
                                'reelUrl': f"https://www.instagram.com/reel/{vid}/",
                                'likes': likes,
                                'category': category,
                                'photo': out_name,
                                'score': float(best_sc),
                                'approved': False
                            }

                            with open(VERIFIED_MANIFEST, 'w') as mf: json.dump(manifest, mf, indent=2)
                            with open(CREATORS_PATH, 'w') as cf: json.dump(list(manifest.values()), cf, indent=2)

                            print(f"  ★ [QUEUED] #{rank_str}: @{handle} -> {likes:,} likes (conf: {best_sc:.1f}%)")
                            break
                    except Exception as e:
                        pass
            except Exception as e:
                print(f"Error on @{handle}: {e}")

        await ctx.close()
    print("\nBatch complete.")

if __name__ == '__main__':
    asyncio.run(run())
