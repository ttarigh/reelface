#!/usr/bin/env python3
import os
import re
import json
import time
import fcntl
import sqlite3
import urllib.request
import ssl
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_MANIFEST = os.path.join(BASE_DIR, 'assets', 'verified_50k_manifest.json')
CREATORS_JSON = os.path.join(BASE_DIR, 'assets', 'creators.json')
WEB_CREATORS = os.path.join(BASE_DIR, 'web', 'assets', 'creators.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'talking_head_photos_100')
BLACKLIST_PATH = os.path.join(BASE_DIR, 'assets', 'rejected_blacklist.json')
LOCK_PATH = os.path.join(BASE_DIR, "assets", "manifest.lock")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
LBF_MODEL = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")

TARGET_PENDING = 28

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("scratch", exist_ok=True)

ctx_ssl = ssl._create_unverified_context()

INDIC_SCRIPT_REGEX = re.compile(r'[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0A80-\u0AFF\u0980-\u09FF\u0A00-\u0A7F\u0D00-\u0D7F]')

REGIONAL_BLOCK_KEYWORDS = [
    'india', 'indian', 'delhi', 'mumbai', 'bangalore', 'punjab', 'desi', 'hindi', 'bhai', 'vines',
    'urdu', 'pakistan', 'kolkata', 'chennai', 'hyderabad', 'bhojpuri', 'malayalam', 'kerala',
    '_in$', 'official_in', 'media_in', 'kushwah', 'sharma', 'singh', 'kumar', 'verma', 'yadav',
    'sahu', 'patel', 'shukla', 'tiwari', 'pandey', 'gupta', 'agarwal', 'mishra', 'khan'
]

GENERAL_BLOCKED_KEYWORDS = [
    'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
    'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
    'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
    'derm', 'dermatolog', 'dr.', 'dr_', 'accent', 'pronounc', 'esl', 'medical', 'md', 'pharma',
    'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german', 'japanese', 'chinese',
    'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language', 'news', 'music', 'singer',
    'radio', 'band', 'guitar', 'vogue', 'hollywood', 'unlocked', 'official', 'memes', 'dailymemes'
]

class FileLock:
    def __enter__(self):
        self.fp = open(LOCK_PATH, "w")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
        except: pass

local_data = threading.local()

def get_models():
    if not hasattr(local_data, 'detector'):
        local_data.detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
        local_data.facemark = cv2.face.createFacemarkLBF()
        local_data.facemark.loadModel(LBF_MODEL)
    return local_data.detector, local_data.facemark

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

def eval_frame(fr):
    h, w = fr.shape[:2]
    if h <= w: return False, 0

    top_bar = fr[:int(h*0.10), :]
    bot_bar = fr[int(h*0.90):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector, facemark = get_models()
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None: return False, 0

    # Filter out tiny background specks/posters (<1.2% area or low confidence)
    valid_faces = [f for f in faces if (f[2]*f[3])/(w*h) >= 0.012 and f[-1] >= 0.70]
    if len(valid_faces) != 1: return False, 0

    primary = valid_faces[0]
    conf = float(primary[-1])
    if conf < 0.78: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Portrait framing: 2.0% to 42.0%
    if area < 0.020 or area > 0.42: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.28: return False, 0
    if fy < int(h*0.04) or fy > int(h*0.65): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 35: return False, 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.30: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.16: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.12 or m_ratio > 0.95: return False, 0

    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

stop_harvest = False

def process_candidate(cand):
    global stop_harvest
    if stop_harvest: return

    vid = cand['video_id']
    cat = cand.get('category', 'general')

    # Quick pre-check before network call
    with FileLock():
        if os.path.exists(BLACKLIST_PATH):
            with open(BLACKLIST_PATH) as bf:
                bld = json.load(bf)
            if vid in set(bld.get('video_ids', [])):
                return
        with open(VERIFIED_MANIFEST) as mf:
            curr_mf = json.load(mf)
        if vid in {v.get('video_id') for v in curr_mf.values()}:
            return
        pending_now = sum(1 for v in curr_mf.values() if not v.get('approved'))
        if pending_now >= TARGET_PENDING:
            stop_harvest = True
            return

    embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
    req = urllib.request.Request(embed_url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })

    try:
        with urllib.request.urlopen(req, timeout=4.0, context=ctx_ssl) as u:
            html = u.read().decode('utf-8', errors='ignore')
    except:
        return

    # 1. Likes check
    real_likes = None
    m_likes = re.search(r'([0-9,.]+)\s*([KkMmBbn]?)\s+likes', html, re.I)
    if m_likes:
        try:
            val = float(m_likes.group(1).replace(',', ''))
            unit = (m_likes.group(2) or '').upper()
            if unit == 'K': val *= 1000
            elif unit == 'M': val *= 1000000
            real_likes = int(val)
        except: pass
    if real_likes is None:
        m_edge = re.search(r'edge_liked_by[\\":\s]+count[\\":\s]+([0-9]+)', html)
        if m_edge: real_likes = int(m_edge.group(1))

    # Use database like count fallback if embed did not list exact likes
    if (real_likes is None or real_likes < 50000) and cand.get('likes') and cand['likes'] >= 50000:
        real_likes = int(cand['likes'])

    if real_likes is None or real_likes < 50000:
        return

    # 2. Handle check
    m_handle = re.search(r'data-ios-link=\"user\?username=([^\"]+)\"', html) or re.search(r'\"username\":\s*\"([^\"]+)\"', html)
    raw_handle = m_handle.group(1).split('&')[0].split('\n')[0].strip() if m_handle else cand.get('handle', '')
    clean_handle = re.sub(r'[^a-zA-Z0-9._-]', '', raw_handle)[:25]
    if not clean_handle or clean_handle.lower() in ['instagram', 'view profile', 'watch on instagram']:
        clean_handle = cand.get('handle', '')
    if not clean_handle: return
    clean_lower = clean_handle.lower()

    if clean_lower.startswith('dr_') or clean_lower.startswith('dr.') or clean_lower.startswith('doctor'):
        return

    if INDIC_SCRIPT_REGEX.search(html):
        return

    if any(b in clean_lower for b in REGIONAL_BLOCK_KEYWORDS):
        return

    if any(b in clean_lower for b in GENERAL_BLOCKED_KEYWORDS):
        return

    # 3. Check Blacklist & Manifest under FileLock
    with FileLock():
        if os.path.exists(BLACKLIST_PATH):
            with open(BLACKLIST_PATH) as bf:
                bld = json.load(bf)
            if vid in set(bld.get('video_ids', [])) or clean_lower in {h.lower() for h in bld.get('handles', [])}:
                return

        with open(VERIFIED_MANIFEST) as mf:
            curr_mf = json.load(mf)
        if clean_lower in {v['handle'].lower() for v in curr_mf.values()} or vid in {v.get('video_id') for v in curr_mf.values()}:
            return

    best_f = None
    best_s = -1.0

    # 4. Check img cover thumbnail
    m_img = re.search(r'class=\"EmbeddedMediaImage\"[^>]*src=\"([^\"]+)\"', html)
    if m_img:
        img_url = m_img.group(1).replace('&amp;', '&')
        try:
            req_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_img, timeout=3.5, context=ctx_ssl) as iu:
                arr = np.frombuffer(iu.read(), dtype=np.uint8)
                fr = cv2.imdecode(arr, -1)
            if fr is not None:
                ok, sc = eval_frame(fr)
                if ok:
                    best_f = fr
                    best_s = sc
        except: pass

    # 5. Fallback: Check video stream frames
    if best_f is None:
        idx = html.find('video_url')
        if idx != -1:
            end_idx = html.find('.mp4', idx)
            if end_idx != -1:
                quote_idx = html.find('\"', end_idx)
                if quote_idx != -1:
                    raw_url = html[idx+len('video_url'):quote_idx]
                    bs = chr(92)
                    v_url = 'https:' + raw_url.split('https:')[-1].replace(bs, '')
                    temp_mp4 = f"scratch/master_{threading.get_ident()}_{vid}.mp4"
                    try:
                        req_v = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req_v, timeout=4.5, context=ctx_ssl) as vu, open(temp_mp4, 'wb') as f:
                            f.write(vu.read())

                        cap = cv2.VideoCapture(temp_mp4)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total > 0:
                            step = max(1, total // 8)
                            for frame_idx in range(max(1, step//2), total, step):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, fr = cap.read()
                                if not ret or fr is None: continue
                                ok, sc = eval_frame(fr)
                                if ok and sc > best_s:
                                    best_s = sc
                                    best_f = fr.copy()
                        cap.release()
                        if os.path.exists(temp_mp4): os.remove(temp_mp4)
                    except:
                        if os.path.exists(temp_mp4): os.remove(temp_mp4)

    if best_f is None:
        return

    # 6. Atomic save under FileLock
    with FileLock():
        with open(VERIFIED_MANIFEST) as mf:
            manifest = json.load(mf)

        if clean_lower in {v['handle'].lower() for v in manifest.values()} or vid in {v.get('video_id') for v in manifest.values()}:
            return

        existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
        next_rank = max(existing_ranks + [0]) + 1
        rank_str = f"{next_rank:04d}"
        out_name = f"{rank_str}_{clean_handle}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, best_f, [cv2.IMWRITE_JPEG_QUALITY, 95])

        reel_url = f"https://www.instagram.com/reel/{vid}/"
        manifest[rank_str] = {
            'rank': rank_str,
            'handle': clean_handle,
            'video_id': vid,
            'reelUrl': reel_url,
            'likes': int(real_likes),
            'category': cat,
            'photo': out_name,
            'score': float(round(best_s, 1)),
            'verified': True,
            'approved': False
        }

        tmp_mf = f"{VERIFIED_MANIFEST}.master_{time.time()}.tmp"
        with open(tmp_mf, 'w') as mf:
            json.dump(manifest, mf, indent=2)
        os.replace(tmp_mf, VERIFIED_MANIFEST)

        creators_list = list(manifest.values())
        tmp_cf = f"{CREATORS_JSON}.master_{time.time()}.tmp"
        with open(tmp_cf, 'w') as cf:
            json.dump(creators_list, cf, indent=2)
        os.replace(tmp_cf, CREATORS_JSON)

        if os.path.exists(os.path.dirname(WEB_CREATORS)):
            tmp_wf = f"{WEB_CREATORS}.master_{time.time()}.tmp"
            with open(tmp_wf, 'w') as wf:
                json.dump(creators_list, wf, indent=2)
            os.replace(tmp_wf, WEB_CREATORS)

        app_total = sum(1 for v in manifest.values() if v.get('approved'))
        pending_total = sum(1 for v in manifest.values() if not v.get('approved'))
        print(f"  ★ [{rank_str}] STAGED ({cat}): @{clean_handle} -> {real_likes:,} REAL LIKES ({reel_url}) | Approved: {app_total}, Pending: {pending_total}/{TARGET_PENDING}", flush=True)

        if pending_total >= TARGET_PENDING:
            print(f"\n>>> [GOAL REACHED] Staged target of {TARGET_PENDING} pending cards! Stopping...", flush=True)
            stop_harvest = True

def main():
    print(f"=== Master Harvester (Targeting {TARGET_PENDING} Pending Cards) ===\n", flush=True)

    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    seen_vids = set(v.get('video_id') for v in manifest.values() if v.get('video_id')) | blacklisted_vids

    candidates = []

    # 1. Load candidates from viral_videos.db (HIGHEST QUALITY, TOP VIRAL CREATORS)
    if os.path.exists("viral_videos.db"):
        conn = sqlite3.connect("viral_videos.db")
        c = conn.cursor()
        c.execute("SELECT creator_id, video_id, like_count FROM videos WHERE like_count >= 50000 ORDER BY like_count DESC")
        for cid, vid, likes in c.fetchall():
            h = cid.replace("ig:", "").replace("@", "").strip().lower()
            if vid not in seen_vids and not any(b in h for b in GENERAL_BLOCKED_KEYWORDS) and not any(b in h for b in REGIONAL_BLOCK_KEYWORDS):
                seen_vids.add(vid)
                candidates.append({'handle': h, 'video_id': vid, 'likes': likes, 'category': 'viral_db'})
        conn.close()

    # 2. Candidate pools from JSON files
    priority_sources = [
        ('candidate_explore_reels.json', None),
        ('candidates_200.json', ['booktok', 'fitnesstrainer', 'sciencefacts', 'softwareengineer']),
        ('fresh_popular_candidates.json', None),
        ('candidates_500.json', ['sciencefacts', 'paleontology', 'ancienthistory', 'voiceactor', 'marinebiology', 'astrophysics', 'softwaredeveloper', 'quantumphysics', 'productdesign', 'woodworkingtips', 'architecturedesign']),
        ('all_pro_candidates.json', None)
    ]

    for fn, tags in priority_sources:
        path = os.path.join(BASE_DIR, fn)
        if not os.path.exists(path): continue
        with open(path) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        for it in items:
            vid = it.get('video_id')
            tag = it.get('tag') or it.get('category') or 'general'
            if tags and tag not in tags: continue
            if vid and vid not in seen_vids:
                seen_vids.add(vid)
                candidates.append({'video_id': vid, 'category': tag})

    print(f"Loaded {len(candidates)} total candidate reels to process.", flush=True)

    current_pending = sum(1 for v in manifest.values() if not v.get('approved'))
    print(f"Current Approved: {sum(1 for v in manifest.values() if v.get('approved'))}, Current Pending: {current_pending}/{TARGET_PENDING}\n", flush=True)

    if current_pending >= TARGET_PENDING:
        print("Target already reached!")
        return

    with ThreadPoolExecutor(max_workers=12) as executor:
        for _ in executor.map(process_candidate, candidates):
            if stop_harvest:
                break

    print("\n=== Master Harvester Finished! ===", flush=True)

if __name__ == '__main__':
    main()
