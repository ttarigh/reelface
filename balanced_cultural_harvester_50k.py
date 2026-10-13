import asyncio
import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
MARKDOWN_OUT = "talking_heads_1000.md"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

# 20 Diverse Cultural Categories of Organic "Talking to Camera with Captions" Reels
# Each category capped at 25 faces max to ensure rich diversity across Instagram culture
CULTURAL_CATEGORIES = {
    "storytime_confessions": [
        "storytime", "relatable", "truestory", "confession", "crazystory", "embarrassing"
    ],
    "dating_relationships": [
        "datingadvice", "relationshiptips", "datingstorytime", "redflags", "modernlove"
    ],
    "fitness_nutrition": [
        "gymtips", "fitnessadvice", "fatlosstips", "nutritionhacks", "formtips"
    ],
    "personal_finance": [
        "moneytips", "personalfinance", "investingtips", "financialfreedom", "budgetingtips"
    ],
    "grwm_lifestyle": [
        "grwm", "beautytips", "skincareroutine", "makeuphacks", "morningroutine"
    ],
    "career_workplace": [
        "careeradvice", "workplacetips", "corporatehumor", "corporatelife", "interviewtips"
    ],
    "booktok_reviews": [
        "booktok", "bookrecommendations", "bookreview", "mustreadbooks", "readersofinstagram"
    ],
    "tech_productivity": [
        "techtips", "iphonetips", "productivityhacks", "coolwebsites", "techhacks"
    ],
    "cooking_foodhacks": [
        "cookinghacks", "cheftips", "foodhacks", "kitchenhacks", "cookingtips"
    ],
    "popculture_film": [
        "popculture", "movietalk", "filmcommentary", "celebritynews", "moviereview"
    ],
    "travel_expat": [
        "traveltips", "travelhacks", "solotravel", "expatlife", "traveladvice"
    ],
    "parenting_family": [
        "parentingadvice", "momlife", "dadlife", "parentinghacks", "parentingtruths"
    ],
    "psychology_wellness": [
        "psychologyfacts", "therapytips", "mentalhealthmatters", "mindsetshift", "attachmentstyles"
    ],
    "fashion_styling": [
        "fashiontips", "stylehacks", "outfitideas", "mensstyle", "fashionadvice"
    ],
    "home_organization": [
        "homehacks", "cleaninghacks", "homeorganization", "amazonfinds", "roommakeover"
    ],
    "automotive_cars": [
        "cartips", "mechanictips", "caradvice", "usedcartips", "drivingtips"
    ],
    "music_voice": [
        "singingtips", "musicfacts", "vocalcoach", "songwritingtips", "musicproducer"
    ],
    "history_trivia": [
        "historyfacts", "interestingfacts", "todayilearned", "historybuff", "obscurefacts"
    ],
    "pov_monologues": [
        "povacting", "actingchallenge", "monologue", "pov", "relatablepov"
    ],
    "motivation_truths": [
        "hardtruths", "discipline", "lifeadvice", "wisdomoftheday", "harshreality"
    ]
}

CATEGORY_CAP = 40  # 40 reels per category to reach 1,000 diverse faces

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def ear(pts):
    r_h1 = np.linalg.norm(pts[37] - pts[41])
    r_h2 = np.linalg.norm(pts[38] - pts[40])
    r_w  = np.linalg.norm(pts[36] - pts[39])
    r_ear = (r_h1 + r_h2) / (2.0 * r_w + 1e-5)
    
    l_h1 = np.linalg.norm(pts[43] - pts[47])
    l_h2 = np.linalg.norm(pts[44] - pts[46])
    l_w  = np.linalg.norm(pts[42] - pts[45])
    l_ear = (l_h1 + l_h2) / (2.0 * l_w + 1e-5)
    
    return (r_ear + l_ear) / 2.0

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    
    # Check 1: Strict Anti-Movie Letterbox / Black Borders
    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0
        
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return False, 0
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area = (fw*fh)/(w*h)
    
    # Check 2: Head & Shoulders / Phone Framing (5.0% to 26.0% of frame)
    # Rejects distant stage shots and tiny background faces
    if area < 0.050 or area > 0.26: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.18: return False, 0
    if fy < int(h*0.06) or fy > int(h*0.48): return False, 0
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 70: return False, 0
    
    # Check 3: Direct Camera Gaze (Looking dead-center into phone camera lens)
    # Rejects comedians looking at live audience and TV actors looking off-camera
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.18: return False, 0
    
    # Check 4: Anti-Cartoon / Anti-Anime / Photorealism
    # Crop central face and measure unique color richness (real cameras have 2500+ unique colors from sensor noise & skin pores)
    face_crop = fr[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if face_crop.size > 0:
        unique_colors = len(np.unique(face_crop.reshape(-1, 3), axis=0))
        if unique_colors < 2500:
            return False, 0 # Blocks flat-shaded cartoons, anime, and 3D renders
            
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    # Eyes open and alert
    eye_ratio = ear(pts)
    if eye_ratio < 0.25: return False, 0
    
    # Mouth actively talking
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.26 or m_ratio > 0.85: return False, 0
    
    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

def download_and_analyze_video(v_src, worker_id, vid, detector, facemark):
    try:
        mp4_path = f"scratch/cult_w{worker_id}_{vid}.mp4"
        req = urllib.request.Request(v_src, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u, open(mp4_path, 'wb') as f:
            f.write(u.read())

        cap = cv2.VideoCapture(mp4_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            if os.path.exists(mp4_path): os.remove(mp4_path)
            return None, -1.0

        step = max(1, total // 35)
        best_f = None
        best_s = -1.0
        for idx in range(10, min(total, 400), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if not ret or fr is None: continue
            ok, sc = eval_frame(fr, detector, facemark)
            if ok and sc > best_s:
                best_s = sc
                best_f = fr.copy()
        cap.release()
        if os.path.exists(mp4_path):
            os.remove(mp4_path)

        return best_f, best_s
    except Exception:
        return None, -1.0

async def worker(worker_id, queue, saved_counter, seen_vids, seen_handles, category_counts, manifest, lock, ctx):
    detector, facemark = create_models()
    page = await ctx.new_page()

    # Blocked keywords in handle (curation, clipping, studio, comedy club accounts)
    blocked_handle_keywords = ['comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel', 'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central']

    while not queue.empty():
        try:
            cand = await queue.get()
        except asyncio.QueueEmpty:
            break

        vid = cand.get('video_id')
        cat = cand.get('category', 'general')

        if not vid or vid in seen_vids:
            try: queue.task_done()
            except ValueError: pass
            continue

        async with lock:
            if saved_counter['count'] >= 1000:
                try: queue.task_done()
                except ValueError: pass
                break
            if category_counts.get(cat, 0) >= CATEGORY_CAP:
                try: queue.task_done()
                except ValueError: pass
                continue

        try:
            embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
            await page.goto(embed_url, wait_until='domcontentloaded', timeout=4000)
            await asyncio.sleep(0.3)

            # Parse verified likes
            text = await page.evaluate("() => document.body.innerText")
            likes = -1
            for line in text.split('\n'):
                m = re.search(r'([0-9,]+)\s+likes', line, re.I)
                if m:
                    likes = int(m.group(1).replace(',', ''))
                    break

            # STRICT 50,000 LIKES HARD GATE
            if likes < 50000:
                continue

            # Extract author
            header_author = await page.evaluate(r'''() => {
                const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*="/p/"], .UsernameText');
                return el ? el.innerText.trim() : '';
            }''')

            clean = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25] if header_author else cand.get('author', f"creator_{vid[:8]}")
            clean_lower = clean.lower()

            # Reject curation/clipping/stage channels
            if any(b in clean_lower for b in blocked_handle_keywords):
                continue

            prefix = clean_lower[:6]
            if prefix in seen_handles:
                continue

            # Get video stream src
            v_src = await page.evaluate(r'''() => {
                const v = document.querySelector('video');
                return v ? v.src : null;
            }''')
            if not v_src:
                try:
                    el = await page.wait_for_selector('video', timeout=1800)
                    v_src = await el.get_attribute('src')
                except Exception:
                    pass
            if not v_src:
                continue

            # Download & run computer vision validation
            best_f, best_s = await asyncio.to_thread(download_and_analyze_video, v_src, worker_id, vid, detector, facemark)

            if best_f is not None:
                async with lock:
                    if saved_counter['count'] < 1000 and category_counts.get(cat, 0) < CATEGORY_CAP and prefix not in seen_handles:
                        saved_counter['count'] += 1
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                        count = saved_counter['count']
                        rank_str = f"{count:04d}"
                        out_name = f"{rank_str}_{clean}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, out_name)
                        cv2.imwrite(out_path, best_f)

                        seen_vids.add(vid)
                        seen_handles.add(prefix)

                        reel_url = f"https://www.instagram.com/reel/{vid}/"
                        manifest[rank_str] = {
                            'rank': rank_str,
                            'handle': clean,
                            'video_id': vid,
                            'reelUrl': reel_url,
                            'likes': likes,
                            'category': cat,
                            'photo': out_name,
                            'score': float(best_s)
                        }

                        with open(VERIFIED_MANIFEST, 'w') as f:
                            json.dump(manifest, f, indent=2)

                        print(f"  [W{worker_id}] ✓ [{rank_str}/1000] [{cat} ({category_counts[cat]}/{CATEGORY_CAP})] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_s:.1f})", flush=True)

        except Exception:
            pass
        finally:
            try: queue.task_done()
            except ValueError: pass

    await page.close()

def run_procrustes_compositing(total_faces):
    print(f"\n=======================================================")
    print(f"RUNNING PROCRUSTEAN COMPOSITING FOR {total_faces} VERIFIED FACES")
    print(f"=======================================================")
    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
        
    creators_list = [manifest[f"{i+1:04d}"] for i in range(total_faces) if f"{i+1:04d}" in manifest]
    CANVAS_W, CANVAS_H = 800, 1000
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    
    aligned_images = []
    canonical_eye_r = np.array([285.0, 400.0], dtype=np.float32)
    canonical_eye_l = np.array([513.0, 397.0], dtype=np.float32)
    dst_eyes = np.array([canonical_eye_r, canonical_eye_l], dtype=np.float32)
    
    for c in creators_list:
        p_path = os.path.join(OUTPUT_DIR, c['photo'])
        if not os.path.exists(p_path): continue
        img = cv2.imread(p_path)
        if img is None: continue
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            f = faces[0]
            re_pt = np.array([f[4], f[5]], dtype=np.float32)
            le_pt = np.array([f[6], f[7]], dtype=np.float32)
            src_eyes = np.array([re_pt, le_pt], dtype=np.float32)
            M = cv2.estimateAffinePartial2D(src_eyes.reshape(1, 2, 2), dst_eyes.reshape(1, 2, 2))[0]
            if M is not None:
                warped = cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                aligned_images.append(warped.astype(np.float32))
                
    if aligned_images:
        avg_img = np.mean(aligned_images, axis=0)
        avg_bgr = np.clip(avg_img, 0, 255).astype(np.uint8)
        comp_name = f"composited_{total_faces}_faces.jpg"
        cv2.imwrite(comp_name, avg_bgr)
        cv2.imwrite("composited_1000_faces.jpg", avg_bgr)
        print(f"  ✓ Saved {comp_name} & updated composited_1000_faces.jpg")

async def collect_url_reels(page, url):
    reels = []
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=4000)
        await asyncio.sleep(0.3)
        links = await page.evaluate(r'''() => {
            return Array.from(document.querySelectorAll('a[href*="/reel/"]')).map(a => a.href.split('?')[0]);
        }''')
        for l in links:
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', l)
            if m: reels.append(m.group(1))
    except Exception:
        pass
    return list(set(reels))

async def main():
    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)

    current_count = len(manifest)
    print(f"=== Starting Balanced Cultural Harvester (Current Pristine: {current_count}/1000) ===")

    seen_vids = {item.get('video_id') for item in manifest.values() if item.get('video_id')}
    seen_handles = {item.get('handle', '')[:6].lower() for item in manifest.values() if item.get('handle')}
    
    # Track category distribution
    category_counts = {}
    for v in manifest.values():
        cat = v.get('category', 'general')
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("Current Category Breakdown:")
    for c, cnt in sorted(category_counts.items()):
        print(f"  {c:25}: {cnt}/{CATEGORY_CAP}")

    saved_counter = {'count': current_count}
    lock = asyncio.Lock()

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        collector_page = await ctx.new_page()

        categories_list = list(CULTURAL_CATEGORIES.keys())
        cat_pointer = 0

        while saved_counter['count'] < 1000:
            active_cats = [c for c in categories_list if category_counts.get(c, 0) < CATEGORY_CAP]
            if not active_cats:
                print("All categories have hit their 25-reel cap! Expanding cap...")
                break

            current_cat = active_cats[cat_pointer % len(active_cats)]
            cat_pointer += 1
            tags = CULTURAL_CATEGORIES[current_cat]

            print(f"\n--- Harvesting Category: [{current_cat}] ({category_counts.get(current_cat, 0)}/{CATEGORY_CAP} saved) ---", flush=True)

            batch_vids = []
            for tag in tags:
                url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                vids = await collect_url_reels(collector_page, url)
                for v in vids:
                    if v not in seen_vids: batch_vids.append({'video_id': v, 'category': current_cat})

            batch_vids = batch_vids[:80] # Process in responsive chunks
            print(f"Queuing {len(batch_vids)} fresh candidates for [{current_cat}] across 8 workers...", flush=True)

            queue = asyncio.Queue()
            for item in batch_vids:
                queue.put_nowait(item)

            num_workers = 8
            workers = [asyncio.create_task(worker(i, queue, saved_counter, seen_vids, seen_handles, category_counts, manifest, lock, ctx)) for i in range(num_workers)]
            await asyncio.gather(*workers)

            if saved_counter['count'] % 50 == 0 or saved_counter['count'] >= 1000:
                run_procrustes_compositing(saved_counter['count'])

        await collector_page.close()
        await ctx.close()

    print(f"\n=======================================================")
    print(f"HARVEST COMPLETE! Total verified faces: {saved_counter['count']}/1000")
    print(f"=======================================================")

    run_procrustes_compositing(saved_counter['count'])

    creators_list = list(manifest.values())
    with open(CREATORS_PATH, 'w') as f:
        json.dump(creators_list, f, indent=2)

    if os.path.exists('web/assets'):
        with open('web/assets/creators.json', 'w') as f:
            json.dump(creators_list, f, indent=2)

    with open(MARKDOWN_OUT, 'w') as out:
        out.write("# Viral Talking-Head Creators Catalog (1000 Faces)\n\n")
        out.write("Total Verified Viral Talking Heads (All >= 50,000 Likes): **1000**\n\n")
        out.write("| Rank | Creator Handle | Category | Reel Link | Verified Likes | Screenshot |\n")
        out.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for c in creators_list:
            r = c['rank']
            h = c['handle']
            cat = c.get('category', 'general')
            link = c['reelUrl']
            likes_fmt = f"{c.get('likes', 0):,}"
            photo = c['photo']
            out.write(f"| {r} | @{h} | {cat} | [Watch Reel]({link}) | **{likes_fmt}** | [{photo}](file:///Users/tarighian/most-viral-face/{OUTPUT_DIR}/{photo}) |\n")

    print(f"Synced {len(creators_list)} creators to {CREATORS_PATH} and {MARKDOWN_OUT}!")

if __name__ == "__main__":
    asyncio.run(main())
