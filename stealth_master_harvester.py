import os
import re
import json
import glob
import asyncio
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
BLACKLIST_PATH = "assets/rejected_blacklist.json"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

CULTURAL_CATEGORIES = {
    "storytime_confessions": ["storytime", "relatable", "truestory", "confession", "crazystory"],
    "dating_relationships": ["datingadvice", "relationshiptips", "datingstorytime", "redflags"],
    "fitness_nutrition": ["gymtips", "fitnessadvice", "fatlosstips", "nutritionhacks"],
    "personal_finance": ["moneytips", "personalfinance", "investingtips", "budgetingtips"],
    "grwm_lifestyle": ["grwm", "beautytips", "skincareroutine", "morningroutine"],
    "career_workplace": ["careeradvice", "workplacetips", "corporatehumor", "interviewtips"],
    "booktok_reviews": ["booktok", "bookrecommendations", "bookreview", "mustreadbooks"],
    "tech_productivity": ["techtips", "iphonetips", "productivityhacks", "coolwebsites"],
    "cooking_foodhacks": ["cookinghacks", "cheftips", "foodhacks", "kitchenhacks"],
    "popculture_film": ["popculture", "movietalk", "filmcommentary", "celebritynews"],
    "travel_expat": ["traveltips", "travelhacks", "solotravel", "expatlife"],
    "parenting_family": ["parentingadvice", "momlife", "dadlife", "parentinghacks"],
    "psychology_wellness": ["psychologyfacts", "therapytips", "mentalhealthmatters", "mindsetshift"],
    "fashion_styling": ["fashiontips", "stylehacks", "outfitideas", "mensstyle"],
    "home_organization": ["homehacks", "cleaninghacks", "homeorganization", "amazonfinds"],
    "automotive_cars": ["cartips", "mechanictips", "caradvice", "usedcartips"],
    "music_voice": ["singingtips", "musicfacts", "vocalcoach", "songwritingtips"],
    "history_trivia": ["historyfacts", "interestingfacts", "todayilearned", "historybuff"],
    "pov_monologues": ["povacting", "actingchallenge", "monologue", "relatablepov"],
    "motivation_truths": ["hardtruths", "discipline", "lifeadvice", "wisdomoftheday"]
}

CATEGORY_CAP = 250

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

    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0

    sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    primary = sorted_faces[0]
    conf = float(primary[-1])
    if conf < 0.82: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Head & shoulders framing (3.0% to 28.0% of frame)
    if area < 0.030 or area > 0.28: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.22: return False, 0
    if fy < int(h*0.05) or fy > int(h*0.58): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 45: return False, 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.20: return False, 0

    face_crop = fr[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if face_crop.size > 0:
        unique_colors = len(np.unique(face_crop.reshape(-1, 3), axis=0))
        if unique_colors < 1600: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.20: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.18 or m_ratio > 0.88: return False, 0

    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

def analyze_video_url(v_url, worker_id, vid, detector, facemark):
    temp_path = f"scratch/stealth_w{worker_id}_{vid}.mp4"
    try:
        req = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u, open(temp_path, 'wb') as f:
            f.write(u.read())

        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            if os.path.exists(temp_path): os.remove(temp_path)
            return None, -1.0

        step = max(1, total // 30)
        best_f = None
        best_s = -1.0
        for idx in range(10, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if not ret or fr is None: continue
            ok, sc = eval_frame(fr, detector, facemark)
            if ok and sc > best_s:
                best_s = sc
                best_f = fr.copy()
        cap.release()
        if os.path.exists(temp_path): os.remove(temp_path)
        return best_f, best_s
    except Exception:
        if os.path.exists(temp_path): os.remove(temp_path)
        return None, -1.0

async def worker(worker_id, queue, lock, browser):
    detector, facemark = create_models()
    context = await browser.new_context(viewport={'width': 600, 'height': 900})
    page = await context.new_page()
    await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")

    blocked_keywords = [
        'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
        'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
        'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
        'derm', 'dermatolog', 'dr.', 'dr_', 'dr', 'accent', 'pronounc', 'esl', 'medical', 'md', 'pharma',
        'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german', 'japanese', 'chinese',
        'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language', 'podcast', 'news', 'music', 'singer',
        'radio', 'band', 'guitar', 'vogue'
    ]

    count = 0

    while not queue.empty():
        try:
            cand = await queue.get()
        except:
            break

        vid = cand.get('video_id')
        cat = cand.get('category', 'general')
        count += 1

        if count % 15 == 0:
            print(f"  [W{worker_id}] Processed {count} items... (queue remaining: {queue.qsize()})", flush=True)

        try:
            embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
            await page.goto(embed_url, wait_until='domcontentloaded', timeout=5000)
            await asyncio.sleep(0.3)

            data = await page.evaluate(r'''() => {
                const links = Array.from(document.querySelectorAll('a'));
                let likes = 0;
                let handle = null;
                for (const a of links) {
                    const m = a.innerText.match(/([0-9,.]+)\s*([KkMmBbn]?)\s+likes/i);
                    if (m) {
                        let val = parseFloat(m[1].replace(/,/g, ''));
                        const unit = (m[2] || '').toUpperCase();
                        if (unit === 'K') val *= 1000;
                        else if (unit === 'M') val *= 1000000;
                        likes = Math.round(val);
                    }
                    if (a.href.includes('utm_source=ig_embed') && !a.href.includes('/reel/') && !a.href.includes('/stories/') && a.innerText.trim().length > 0 && !a.innerText.includes('View') && !a.innerText.includes('more')) {
                        handle = a.innerText.trim();
                    }
                }
                const v = document.querySelector('video');
                const img = document.querySelector('.EmbeddedMediaImage');
                return { likes, handle, video_src: v ? v.src : null, img_src: img ? img.src : null };
            }''')

            likes = data.get('likes', 0)
            if likes < 25000:
                continue

            raw_handle = data.get('handle') or f"creator_{vid[:8]}"
            clean = re.sub(r'[^a-zA-Z0-9._-]', '', raw_handle.split('\n')[0].strip())[:25]
            clean_lower = clean.lower()

            if any(b in clean_lower for b in blocked_keywords):
                continue

            with open(VERIFIED_MANIFEST) as mf:
                current_manifest = json.load(mf)
            existing_handles = {v['handle'].lower() for v in current_manifest.values()}
            if clean_lower in existing_handles:
                continue

            best_f = None
            best_s = -1.0

            # 1. Try video stream
            v_src = data.get('video_src')
            if v_src:
                best_f, best_s = analyze_video_url(v_src, worker_id, vid, detector, facemark)

            # 2. Try raw clean image source directly (no browser overlay/play button)
            img_src = data.get('img_src')
            if best_f is None and img_src:
                try:
                    req = urllib.request.Request(img_src, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=5.0, context=ctx_ssl) as u:
                        arr = np.frombuffer(u.read(), dtype=np.uint8)
                        fr = cv2.imdecode(arr, -1)
                    if fr is not None:
                        ok, sc = eval_frame(fr, detector, facemark)
                        if ok:
                            best_f = fr
                            best_s = sc
                except: pass

            # 3. Fallback to direct element screenshot if needed
            if best_f is None:
                img_el = await page.query_selector('.EmbeddedMediaImage')
                if img_el:
                    img_bytes = await img_el.screenshot()
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    fr = cv2.imdecode(arr, -1)
                    if fr is not None:
                        ok, sc = eval_frame(fr, detector, facemark)
                        if ok:
                            best_f = fr
                            best_s = sc

            if best_f is None:
                continue

            # Thread-safe save
            async with lock:
                with open(VERIFIED_MANIFEST) as mf:
                    manifest = json.load(mf)

                if os.path.exists(BLACKLIST_PATH):
                    with open(BLACKLIST_PATH) as bf:
                        bl_data = json.load(bf)
                        current_bl_handles = {h.lower() for h in bl_data.get("handles", [])}
                        current_bl_vids = set(bl_data.get("video_ids", []))
                    if vid in current_bl_vids or clean_lower in current_bl_handles:
                        continue

                for v in manifest.values():
                    if v.get('video_id') == vid or v.get('handle', '').lower() == clean_lower:
                        continue

                cat_count = sum(1 for v in manifest.values() if v.get('category') == cat)
                if cat_count >= CATEGORY_CAP:
                    continue

                existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
                next_rank = max(existing_ranks + [0]) + 1
                rank_str = f"{next_rank:04d}"
                out_name = f"{rank_str}_{clean}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, best_f)

                reel_url = f"https://www.instagram.com/reel/{vid}/"
                manifest[rank_str] = {
                    'rank': rank_str,
                    'handle': clean,
                    'video_id': vid,
                    'reelUrl': reel_url,
                    'likes': likes,
                    'category': cat,
                    'photo': out_name,
                    'score': float(best_s),
                    'approved': False
                }

                with open(VERIFIED_MANIFEST, 'w') as mf:
                    json.dump(manifest, mf, indent=2)

                with open(CREATORS_PATH, 'w') as cf:
                    json.dump(list(manifest.values()), cf, indent=2)

                print(f"  [W{worker_id}] ✓ [{rank_str}/1000] [{cat} ({cat_count + 1}/{CATEGORY_CAP})] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_s:.1f})", flush=True)

        except:
            pass

    await context.close()

async def main():
    print("=== Launching Stealth Master Harvester (Bypasses Login Walls & Rate Limits) ===")

    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
    print(f"Current Verified Faces: {len(manifest)}/1000")

    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    blacklisted_handles = {h.lower() for h in bl_data.get("handles", [])}
    print(f"Loaded {len(blacklisted_handles)} blacklisted handles and {len(blacklisted_vids)} blacklisted video IDs.")

    cached_candidates = []
    seen_cand = set(manifest[k].get('video_id') for k in manifest if manifest[k].get('video_id')) | blacklisted_vids

    priority_files = [
        'fresh_popular_candidates.json',
        'assets/recovered_reels_manifest.json',
        'candidate_explore_reels.json',
        'candidates_500.json',
        'candidates_200.json',
        'broadcast_candidates.json',
        'scratch/live_explore_reels.json',
        'scratch/reserve_candidates.json',
        'talking_heads_100_manifest.json',
        'speakers.json',
        'keynote.json'
    ]
    for jf in priority_files:
        if not os.path.exists(jf): continue
        try:
            with open(jf) as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
                for it in items:
                    v = None
                    if isinstance(it, dict):
                        v = it.get('video_id')
                        if not v and it.get('reelUrl'):
                            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', it['reelUrl'])
                            if m: v = m.group(1)
                    elif isinstance(it, str):
                        m = re.search(r'/reel/([a-zA-Z0-9_-]+)', it)
                        if m: v = m.group(1)
                        elif len(it) == 11 and '/' not in it:
                            v = it
                    if v and v not in seen_cand:
                        seen_cand.add(v)
                        cached_candidates.append(v)
        except: pass

    print(f"Queuing {len(cached_candidates)} priority organic viral reels into stealth pipeline...")

    categories = list(CULTURAL_CATEGORIES.keys())
    queue = asyncio.Queue()
    for i, vid in enumerate(cached_candidates):
        cat = categories[i % len(categories)]
        queue.put_nowait({'video_id': vid, 'category': cat})

    lock = asyncio.Lock()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )

        num_workers = 6
        workers = [asyncio.create_task(worker(i, queue, lock, browser)) for i in range(num_workers)]
        await asyncio.gather(*workers)
        await browser.close()

    print("=== Stealth Harvester Completed Run! ===")

if __name__ == "__main__":
    asyncio.run(main())
