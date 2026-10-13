import os
import re
import json
import time
import urllib.request
import ssl
import cv2
import numpy as np

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
BLACKLIST_PATH = "assets/rejected_blacklist.json"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

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

    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) == 0: return False, 0

    sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    primary = sorted_faces[0]
    conf = float(primary[-1])
    if conf < 0.80: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Head & shoulders framing (3.0% to 28.0% of frame)
    if area < 0.030 or area > 0.28: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.22: return False, 0
    if fy < int(h*0.04) or fy > int(h*0.58): return False, 0

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
        if unique_colors < 1500: return False, 0

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

def main():
    print("=== Launching Proven Viral Creator Harvester ===")
    detector, facemark = create_models()

    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
    print(f"Current Verified Faces: {len(manifest)}/1000")

    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    blacklisted_handles = {h.lower() for h in bl_data.get("handles", [])}

    blocked_keywords = [
        'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
        'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
        'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
        'derm', 'dermatolog', 'dr.', 'dr_', 'dr', 'accent', 'pronounc', 'esl', 'medical', 'md', 'pharma',
        'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german', 'japanese', 'chinese',
        'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language'
    ]

    with open('assets/recovered_reels_manifest.json') as f:
        d = json.load(f)

    items = list(d.values()) if isinstance(d, dict) else d
    candidates = []
    seen_cand = set(manifest[k].get('video_id') for k in manifest if manifest[k].get('video_id')) | blacklisted_vids

    for it in items:
        url = it.get('reelUrl', '')
        m = re.search(r'/reel/([a-zA-Z0-9_-]+)', url)
        if not m: continue
        vid = m.group(1)
        handle = it.get('handle', '').lower()
        if vid in seen_cand or handle in blacklisted_handles: continue
        if any(b in handle for b in blocked_keywords): continue
        candidates.append((vid, handle, it.get('category', 'general')))

    print(f"Queued {len(candidates)} proven viral candidates...")

    for i, (vid, raw_h, cat) in enumerate(candidates):
        with open(VERIFIED_MANIFEST) as f:
            manifest = json.load(f)

        if len(manifest) >= 1000:
            print("=== Reached 1,000 Verified Faces! ===")
            break

        url = f"https://r.jina.ai/https://www.instagram.com/reel/{vid}/embed/"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        try:
            with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u:
                md = u.read().decode('utf-8')

            m_likes = re.search(r'\[([0-9,]+)\s+likes\]', md)
            if not m_likes: m_likes = re.search(r'([0-9,]+)\s+likes', md)
            if not m_likes:
                time.sleep(3.2)
                continue

            likes = int(m_likes.group(1).replace(',', ''))
            if likes < 50000:
                time.sleep(3.2)
                continue

            m_handle = re.search(r'\[([a-zA-Z0-9._-]+)\]\(https://www\.instagram\.com/[a-zA-Z0-9._-]+/\?utm_source=ig_embed', md)
            clean = m_handle.group(1) if m_handle else raw_h
            clean = re.sub(r'[^a-zA-Z0-9._-]', '', clean.split('\n')[0].strip())[:25]
            clean_lower = clean.lower()

            if clean_lower in blacklisted_handles or any(b in clean_lower for b in blocked_keywords):
                time.sleep(3.2)
                continue

            existing_handles = {v['handle'].lower() for v in manifest.values()}
            if clean_lower in existing_handles or clean_lower[:6] in {h[:6] for h in existing_handles}:
                time.sleep(3.2)
                continue

            # Check cover images
            imgs = re.findall(r'!\[[^\]]*\]\((https://scontent[^\)]+)\)', md)
            best_fr = None
            best_sc = -1.0

            for img_url in imgs:
                if 's100x100' in img_url or 's150x150' in img_url: continue
                try:
                    r_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(r_img, timeout=6.0, context=ctx_ssl) as iu:
                        arr = np.asarray(bytearray(iu.read()), dtype=np.uint8)
                        fr = cv2.imdecode(arr, -1)
                    if fr is not None and fr.shape[0] >= 300 and fr.shape[1] >= 250:
                        ok, sc = eval_frame(fr, detector, facemark)
                        if ok and sc > best_sc:
                            best_sc = sc
                            best_fr = fr.copy()
                except: pass

            if best_fr is None:
                time.sleep(3.2)
                continue

            # Save confirmed creator
            with open(VERIFIED_MANIFEST) as mf:
                manifest = json.load(mf)

            existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
            next_rank = max(existing_ranks + [0]) + 1
            rank_str = f"{next_rank:04d}"
            out_name = f"{rank_str}_{clean}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, best_fr)

            reel_url = f"https://www.instagram.com/reel/{vid}/"
            manifest[rank_str] = {
                'rank': rank_str,
                'handle': clean,
                'video_id': vid,
                'reelUrl': reel_url,
                'likes': likes,
                'category': cat,
                'photo': out_name,
                'score': float(best_sc),
                'approved': False
            }

            with open(VERIFIED_MANIFEST, 'w') as mf:
                json.dump(manifest, mf, indent=2)

            with open(CREATORS_PATH, 'w') as cf:
                json.dump(list(manifest.values()), cf, indent=2)

            print(f"  ✓ [{rank_str}/1000] [{cat}] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_sc:.1f})", flush=True)

        except Exception:
            pass

        time.sleep(3.2)

if __name__ == "__main__":
    main()
