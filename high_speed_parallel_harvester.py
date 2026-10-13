import os
import re
import json
import glob
import time
import urllib.request
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import yt_dlp
from playwright.sync_api import sync_playwright

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

CATEGORY_CAP = 40
manifest_lock = threading.Lock()

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

    # Letterbox check (anti-movie)
    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) == 0: return False, 0

    # Primary foreground face
    sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    primary = sorted_faces[0]
    conf = float(primary[-1])
    if conf < 0.88: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Head & shoulders framing (5.0% to 26.0% of frame)
    if area < 0.050 or area > 0.26: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.20: return False, 0
    if fy < int(h*0.06) or fy > int(h*0.50): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 65: return False, 0

    # Direct camera gaze
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.18: return False, 0

    # Photorealism check: real human skin has 2,000+ unique colors in crop
    face_crop = fr[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if face_crop.size > 0:
        unique_colors = len(np.unique(face_crop.reshape(-1, 3), axis=0))
        if unique_colors < 2000: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.24: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.25 or m_ratio > 0.85: return False, 0

    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

def analyze_video_stream(v_url, thread_id, vid, detector, facemark):
    temp_path = f"scratch/stream_t{thread_id}_{vid}.mp4"
    try:
        req = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u, open(temp_path, 'wb') as f:
            f.write(u.read(3500000)) # Read first 3.5MB for fast scanning

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

def process_candidate(task_tuple):
    vid, category, thread_id, detector, facemark, blacklisted_vids, blacklisted_handles, blocked_keywords = task_tuple

    if vid in blacklisted_vids: return False

    reel_url = f"https://www.instagram.com/reel/{vid}/"
    ydl_opts = {'quiet': True, 'no_warnings': True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(reel_url, download=False)
            if not info: return False
            likes = info.get('like_count', 0)
            if likes is None or likes < 50000:
                return False

            uploader = info.get('uploader') or f"creator_{vid[:8]}"
            clean = re.sub(r'[^a-zA-Z0-9._-]', '', uploader)[:25]
            clean_lower = clean.lower()

            if clean_lower in blacklisted_handles: return False
            if any(b in clean_lower for b in blocked_keywords): return False

            formats = [f for f in info.get('formats', []) if f.get('vcodec') != 'none']
            if not formats: return False
            v_url = formats[-1]['url']

            best_f, best_s = analyze_video_stream(v_url, thread_id, vid, detector, facemark)
            if best_f is None: return False

            # Thread-safe atomic write to manifest and disk
            with manifest_lock:
                with open(VERIFIED_MANIFEST) as mf:
                    manifest = json.load(mf)

                # Re-check blacklist from disk
                if os.path.exists(BLACKLIST_PATH):
                    with open(BLACKLIST_PATH) as bf:
                        bl_data = json.load(bf)
                        current_bl_handles = {h.lower() for h in bl_data.get("handles", [])}
                        current_bl_vids = set(bl_data.get("video_ids", []))
                    if vid in current_bl_vids or clean_lower in current_bl_handles:
                        return False

                # Check if already added
                for v in manifest.values():
                    if v.get('video_id') == vid or v.get('handle', '').lower() == clean_lower:
                        return False

                # Check category cap
                cat_count = sum(1 for v in manifest.values() if v.get('category') == category)
                if cat_count >= CATEGORY_CAP:
                    return False

                existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
                next_rank = max(existing_ranks + [0]) + 1
                rank_str = f"{next_rank:04d}"
                out_name = f"{rank_str}_{clean}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, best_f)

                manifest[rank_str] = {
                    'rank': rank_str,
                    'handle': clean,
                    'video_id': vid,
                    'reelUrl': reel_url,
                    'likes': likes,
                    'category': category,
                    'photo': out_name,
                    'score': float(best_s),
                    'approved': False
                }

                with open(VERIFIED_MANIFEST, 'w') as mf:
                    json.dump(manifest, mf, indent=2)

                with open(CREATORS_PATH, 'w') as cf:
                    json.dump(list(manifest.values()), cf, indent=2)

                print(f"  [T{thread_id}] ✓ [{rank_str}/1000] [{category} ({cat_count + 1}/{CATEGORY_CAP})] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_s:.1f})", flush=True)
                return True

    except Exception:
        return False

def harvest_explore_reels(page, tag, num_scrolls=4):
    vids = []
    try:
        url = f"https://www.instagram.com/explore/tags/{tag}/"
        page.goto(url, wait_until='domcontentloaded', timeout=5000)
        for _ in range(num_scrolls):
            page.evaluate("window.scrollBy(0, 1000)")
            page.wait_for_timeout(300)
        links = page.evaluate("() => Array.from(document.querySelectorAll('a[href*=\"/reel/\"]')).map(a => a.href.split('?')[0])")
        for l in links:
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', l)
            if m: vids.append(m.group(1))
    except: pass
    return vids

def main():
    print("=== Launching High-Speed Parallel Harvester (Target: 1,000 Verified Faces) ===")

    # 1. Load active manifest
    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
    print(f"Current Verified Faces: {len(manifest)}/1000")

    # 2. Load permanent blacklist
    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    blacklisted_handles = {h.lower() for h in bl_data.get("handles", [])}

    # Add old_unverified directory handles
    for p in glob.glob("old_unverified/*.jpg"):
        h = os.path.basename(p).replace(".jpg", "").split("_", 1)[-1].lower()
        blacklisted_handles.add(h)

    print(f"Loaded {len(blacklisted_handles)} blacklisted handles and {len(blacklisted_vids)} blacklisted video IDs.")

    blocked_keywords = ['comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel', 'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central', 'songs', 'status', 'cinema', 'theatre', 'film']

    # 3. Collect all 3,000+ candidate reels from repository
    cached_candidates = []
    seen_cand = set(manifest[k].get('video_id') for k in manifest if manifest[k].get('video_id')) | blacklisted_vids

    for jf in glob.glob('*.json') + glob.glob('scratch/*.json'):
        if 'verified' in jf or 'blacklist' in jf or 'creators' in jf: continue
        try:
            with open(jf) as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
                for it in items:
                    v = it.get('video_id')
                    if not v and it.get('reelUrl'):
                        m = re.search(r'/reel/([a-zA-Z0-9_-]+)', it['reelUrl'])
                        if m: v = m.group(1)
                    if v and v not in seen_cand:
                        seen_cand.add(v)
                        cached_candidates.append(v)
        except: pass

    print(f"Loaded {len(cached_candidates)} pre-cached candidate reels ready for parallel evaluation!")

    # 4. Initialize thread-local models for 10 worker threads
    worker_models = {}
    for tid in range(10):
        worker_models[tid] = create_models()

    # 5. Process candidate pool with 10 concurrent threads
    categories = list(CULTURAL_CATEGORIES.keys())
    cat_idx = 0

    tasks = []
    for i, vid in enumerate(cached_candidates):
        tid = i % 10
        det, fm = worker_models[tid]
        cat = categories[cat_idx % len(categories)]
        cat_idx += 1
        tasks.append((vid, cat, tid, det, fm, blacklisted_vids, blacklisted_handles, blocked_keywords))

    print(f"Executing {len(tasks)} candidate evaluations across 10 concurrent parallel threads...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(process_candidate, tasks):
            # Check progress
            with manifest_lock:
                if len(manifest) >= 1000:
                    print("Reached 1,000 verified faces milestone! Exiting...")
                    break

    # 6. Live explore tag expansion if more faces are needed
    if len(manifest) < 1000:
        print("\n--- Live Hashtag Exploration for Remaining Faces ---")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for cat, tags in CULTURAL_CATEGORIES.items():
                if len(manifest) >= 1000: break
                for tag in tags:
                    live_vids = harvest_explore_reels(page, tag, num_scrolls=4)
                    fresh_vids = [v for v in live_vids if v not in seen_cand]
                    if not fresh_vids: continue

                    live_tasks = []
                    for i, vid in enumerate(fresh_vids):
                        seen_cand.add(vid)
                        tid = i % 10
                        det, fm = worker_models[tid]
                        live_tasks.append((vid, cat, tid, det, fm, blacklisted_vids, blacklisted_handles, blocked_keywords))

                    with ThreadPoolExecutor(max_workers=10) as executor:
                        list(executor.map(process_candidate, live_tasks))

            browser.close()

    print(f"=== Harvesting Complete! Total in Database: {len(manifest)} faces ===")

if __name__ == "__main__":
    main()
