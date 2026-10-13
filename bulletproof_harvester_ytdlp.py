import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
import yt_dlp
from playwright.sync_api import sync_playwright

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
MARKDOWN_OUT = "talking_heads_1000.md"
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

    # Sort faces by area descending - primary foreground face
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

def analyze_video_url(v_url, detector, facemark, temp_path):
    try:
        req = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10.0, context=ctx_ssl) as u, open(temp_path, 'wb') as f:
            f.write(u.read())

        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            if os.path.exists(temp_path): os.remove(temp_path)
            return None, -1.0

        step = max(1, total // 35)
        best_f = None
        best_s = -1.0
        # Scan entire duration of the video
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

def main():
    detector, facemark = create_models()
    ydl_opts = {'quiet': True, 'no_warnings': True}
    ydl = yt_dlp.YoutubeDL(ydl_opts)

    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)

    # Load permanent rejection blacklist
    rejected_blacklist_path = "assets/rejected_blacklist.json"
    blacklisted_vids = set()
    blacklisted_handles = set()
    if os.path.exists(rejected_blacklist_path):
        with open(rejected_blacklist_path) as bf:
            bldata = json.load(bf)
            blacklisted_vids = set(bldata.get("video_ids", []))
            blacklisted_handles = {h.lower() for h in bldata.get("handles", [])}

    import glob
    old_files = glob.glob("old_unverified/*.jpg")
    for op in old_files:
        h = os.path.basename(op).replace(".jpg", "").split("_", 1)[-1].lower()
        blacklisted_handles.add(h)

    # Permanent memory: active + all rejected/blacklisted
    seen_vids = {item.get('video_id') for item in manifest.values() if item.get('video_id')} | blacklisted_vids
    seen_handles = {item.get('handle', '')[:6].lower() for item in manifest.values() if item.get('handle')} | {h[:6] for h in blacklisted_handles}

    print(f"Loaded {len(blacklisted_handles)} blacklisted handles and {len(blacklisted_vids)} blacklisted video IDs (these will NEVER be re-harvested)!")

    category_counts = {}
    for v in manifest.values():
        c = v.get('category', 'general')
        category_counts[c] = category_counts.get(c, 0) + 1

    print(f"=== Starting Bulletproof Harvester with yt_dlp (Active: {len(manifest)}/1000) ===")

    blocked_keywords = ['comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel', 'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central']

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        categories_list = list(CULTURAL_CATEGORIES.keys())
        cat_idx = 0

        while len(manifest) < 1000:
            active_cats = [c for c in categories_list if category_counts.get(c, 0) < CATEGORY_CAP]
            if not active_cats:
                print("All categories capped! Finishing...")
                break

            current_cat = active_cats[cat_idx % len(active_cats)]
            cat_idx += 1
            tags = CULTURAL_CATEGORIES[current_cat]

            print(f"\n--- Category [{current_cat}] ({category_counts.get(current_cat, 0)}/{CATEGORY_CAP} saved) ---", flush=True)

            candidate_vids = []
            for tag in tags:
                try:
                    url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                    page.goto(url, wait_until='domcontentloaded', timeout=5000)
                    page.wait_for_timeout(400)
                    links = page.evaluate("() => Array.from(document.querySelectorAll('a[href*=\"/reel/\"]')).map(a => a.href.split('?')[0])")
                    for l in links:
                        m = re.search(r'/reel/([a-zA-Z0-9_-]+)', l)
                        if m and m.group(1) not in seen_vids:
                            candidate_vids.append(m.group(1))
                except Exception:
                    pass

            print(f"Found {len(candidate_vids)} fresh candidate reels for [{current_cat}]. Evaluating with yt_dlp...", flush=True)

            for vid in candidate_vids:
                if len(manifest) >= 1000: break
                if category_counts.get(current_cat, 0) >= CATEGORY_CAP: break
                if vid in seen_vids: continue

                reel_url = f"https://www.instagram.com/reel/{vid}/"
                try:
                    info = ydl.extract_info(reel_url, download=False)
                    if not info: continue
                    likes = info.get('like_count', 0)
                    if likes is None or likes < 50000:
                        continue

                    uploader = info.get('uploader') or f"creator_{vid[:8]}"
                    clean = re.sub(r'[^a-zA-Z0-9._-]', '', uploader)[:25]
                    clean_lower = clean.lower()

                    if clean_lower in blacklisted_handles:
                        continue

                    if any(b in clean_lower for b in blocked_keywords):
                        continue

                    prefix = clean_lower[:6]
                    if prefix in seen_handles:
                        continue

                    formats = [f for f in info.get('formats', []) if f.get('vcodec') != 'none']
                    if not formats: continue
                    v_url = formats[-1]['url']

                    temp_path = f"scratch/cult_{vid}.mp4"
                    best_f, best_s = analyze_video_url(v_url, detector, facemark, temp_path)

                    if best_f is not None:
                        # Always reload manifest and blacklist from disk fresh so user deletions are never overwritten
                        if os.path.exists(VERIFIED_MANIFEST):
                            with open(VERIFIED_MANIFEST) as mf:
                                manifest = json.load(mf)
                        if os.path.exists(rejected_blacklist_path):
                            with open(rejected_blacklist_path) as bf:
                                bldata = json.load(bf)
                                blacklisted_handles = {h.lower() for h in bldata.get("handles", [])}
                                blacklisted_vids = set(bldata.get("video_ids", []))

                        if vid in blacklisted_vids or clean_lower in blacklisted_handles:
                            continue

                        existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
                        next_rank = max(existing_ranks + [0]) + 1
                        rank_str = f"{next_rank:04d}"
                        out_name = f"{rank_str}_{clean}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, out_name)
                        cv2.imwrite(out_path, best_f)

                        seen_vids.add(vid)
                        seen_handles.add(prefix)
                        category_counts[current_cat] = category_counts.get(current_cat, 0) + 1

                        manifest[rank_str] = {
                            'rank': rank_str,
                            'handle': clean,
                            'video_id': vid,
                            'reelUrl': reel_url,
                            'likes': likes,
                            'category': current_cat,
                            'photo': out_name,
                            'score': float(best_s),
                            'approved': False
                        }

                        with open(VERIFIED_MANIFEST, 'w') as f:
                            json.dump(manifest, f, indent=2)

                        with open(CREATORS_PATH, 'w') as f:
                            json.dump(list(manifest.values()), f, indent=2)

                        print(f"  ✓ [{rank_str}/1000] [{current_cat} ({category_counts[current_cat]}/{CATEGORY_CAP})] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_s:.1f})", flush=True)

                except Exception as e:
                    pass

        browser.close()

if __name__ == "__main__":
    main()
