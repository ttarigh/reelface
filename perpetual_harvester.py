import asyncio
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
import os
import time
from playwright.async_api import async_playwright

ctx_ssl = ssl._create_unverified_context()
detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar.onnx', '', (320, 320), score_threshold=0.5)

OUTPUT_DIR = 'talking_head_photos_100'
VERIFIED_MANIFEST = 'assets/verified_50k_manifest.json'
CREATORS_PATH = 'assets/creators.json'
BLACKLIST_PATH = 'assets/rejected_blacklist.json'

TARGETS = [
    ('kallmekris', 'comedy_monologue'),
    ('tinx', 'dating_lifestyle_monologue'),
    ('taraswrld', 'storytime_rant'),
    ('brookeschofield', 'storytime'),
    ('katieflorence', 'comedy_monologue'),
    ('morgannn_wilson', 'storytime'),
    ('clarkepeoples', 'storytime_nyc'),
    ('chloeshih', 'tech_career_chat'),
    ('leenda_dong', 'comedy_acting'),
    ('saraecheagaray', 'acting_monologue'),
    ('sarati', 'storytime'),
    ('reesteees', 'comedy_monologue'),
    ('juliabrookss', 'storytime'),
    ('violetclara', 'storytime'),
    ('victoriabrowne', 'mental_health_chat'),
    ('maryblove', 'lifestyle_chat'),
    ('kenzieelizabeth', 'lifestyle_podcast_chat'),
    ('madelynn_pack', 'fashion_lifestyle_chat'),
    ('theannareid', 'storytime'),
    ('abbymorgan', 'lifestyle_storytime'),
    ('peruseproject', 'booktok'),
    ('jessethereader', 'booktok'),
    ('jackbenedwards', 'booktok'),
    ('uncarley', 'booktok'),
    ('claudia_bouza', 'booktok'),
    ('lexi.m.hidalgo', 'lifestyle_chat'),
    ('piratesoftware', 'tech_gaming_chat'),
    ('codingwithlewis', 'tech_explainer'),
    ('internetvin', 'tech_commentary'),
    ('alexcancode', 'tech_career'),
    ('brennan.elliott', 'tech_career'),
    ('alexeysay', 'tech_career'),
    ('tiffany.janzen', 'tech_career'),
    ('cleverprogrammer', 'tech_career'),
    ('developedbyed', 'tech_career'),
    ('amigoscode', 'tech_career'),
    ('joma', 'tech_comedy'),
    ('meredithduxbury', 'grwm_beauty'),
    ('courtneycahoon', 'grwm_beauty'),
    ('emiliekiser', 'lifestyle_storytime'),
    ('avani', 'lifestyle_chat'),
    ('charlidamelio', 'lifestyle_chat'),
    ('dixiedamelio', 'lifestyle_chat'),
    ('addisonraee', 'lifestyle_chat'),
    ('olivia.ponton', 'lifestyle_chat'),
    ('hannahmeloche', 'lifestyle_vlog'),
    ('summer_mckeen', 'lifestyle_vlog'),
    ('sydney_adams_', 'lifestyle_vlog'),
    ('claire_hogle', 'sports_lifestyle'),
    ('gabby_murrayy', 'lifestyle_vlog')
]

BLOCKED_KEYWORDS = [
    'song', 'singing', 'movie clip', 'cinema', 'trailer', 'stand up',
    'podcast desk', 'hospital', 'doctor', 'clinic', 'dentist', 'surgery'
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

async def harvest_loop():
    print("[DAEMON STARTED] Perpetual Elite Harvester running...")
    
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        while True:
            with open(VERIFIED_MANIFEST) as f: manifest = json.load(f)
            existing_handles = {v['handle'].lower() for v in manifest.values()}

            blacklisted = set()
            if os.path.exists(BLACKLIST_PATH):
                with open(BLACKLIST_PATH) as bf:
                    bl = json.load(bf)
                    blacklisted = {h.lower() for h in bl.get('handles', [])}

            active_targets = [t for t in TARGETS if t[0].lower() not in existing_handles and t[0].lower() not in blacklisted]
            print(f"\n--- Starting Cycle with {len(active_targets)} remaining targets ---")

            if not active_targets:
                print("All targets currently satisfied or reviewed. Sleeping 60s...")
                await asyncio.sleep(60)
                continue

            for handle, category in active_targets:
                # Re-verify in case user took action
                with open(VERIFIED_MANIFEST) as f: manifest = json.load(f)
                if handle.lower() in {v['handle'].lower() for v in manifest.values()}:
                    continue

                print(f"\n[Scraping Profile] @{handle} ({category})...")
                try:
                    await page.goto(f"https://www.instagram.com/{handle}/reels/", wait_until='networkidle', timeout=12000)
                    content = await page.content()
                    reels = list(dict.fromkeys(re.findall(r'/reel/([a-zA-Z0-9_-]+)', content)))
                    print(f"  Found {len(reels)} reel IDs for @{handle}")

                    for vid in reels[:8]:
                        await asyncio.sleep(0.4) # Rate-limit protection for Jina
                        url = f"https://r.jina.ai/https://www.instagram.com/reel/{vid}/embed/"
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        try:
                            with urllib.request.urlopen(req, timeout=6.0, context=ctx_ssl) as u:
                                md = u.read().decode('utf-8')

                            m_likes = re.search(r'\[([0-9,]+)\s+likes\]', md)
                            if not m_likes: m_likes = re.search(r'([0-9,]+)\s+likes', md)
                            likes = int(m_likes.group(1).replace(',', '')) if m_likes else 0
                            if likes < 50000:
                                continue

                            # Check keywords
                            if any(w in md.lower()[:400] for w in BLOCKED_KEYWORDS):
                                continue

                            # Extract frame
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
                                # Append to manifest
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
                                break # Move to next creator once queued
                        except Exception as e:
                            pass
                except Exception as e:
                    print(f"Error on @{handle}: {e}")

            print("Cycle finished. Sleeping 15s before next pass...")
            await asyncio.sleep(15)

if __name__ == '__main__':
    asyncio.run(harvest_loop())
