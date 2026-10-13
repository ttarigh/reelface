import os
import re
import json
import time
import urllib.request
import ssl
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

ctx_ssl = ssl._create_unverified_context()
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.5)

user_rejected_handles = {
    'everydayherox', 'automobileguided', 'sylviapalth', 'artistswithoutautotune', 'colinebergeon',
    'italianmattersmargheritab', 'mr.usaenglish', 'rebeccarose', 'marion', 'careersbychris',
    'dermatologysurgeon', 'dermangelo', 'dripshitajohri', 'alharamaesthetics', 'yours_dermatologically_',
    'english_with_luna1', 'jongnodaek', 'melvinboie', 'annabelledoesvoices', 'maia.isabella', 'timeless stories'
}

blocked_keywords = [
    'doctor', 'derm', 'clinic', 'dentist', 'surgery', 'english', 'spanish', 'french',
    'accent', 'pronounc', 'hospital', 'physio', 'chiro', 'nurse', 'esl', 'dr.', 'dr_', 'dr',
    'healthcare', 'medical', 'pharma', 'scrubs'
]

def loosened_eval(fr):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) == 0: return False, 0
    f = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
    fx, fy, fw, fh = [int(v) for v in f[:4]]
    area = (fw*fh)/(w*h)
    conf = float(f[-1])
    if area < 0.035: return False, 0
    if conf < 0.65: return False, 0
    return True, conf * 100

def get_candidates():
    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    existing_handles = {v['handle'].lower() for v in manifest.values()}
    existing_vids = {v.get('video_id') for v in manifest.values() if v.get('video_id')}

    candidates = []
    seen = set(existing_vids)

    # 1. Load from recovered_reels_manifest.json
    if os.path.exists('assets/recovered_reels_manifest.json'):
        with open('assets/recovered_reels_manifest.json') as f:
            rec = json.load(f)
        items = list(rec.values()) if isinstance(rec, dict) else rec
        for it in items:
            h = (it.get('handle') or '').lower()
            url = it.get('reelUrl', '')
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', url)
            if not m: continue
            vid = m.group(1)
            if vid in seen or h in existing_handles or h in user_rejected_handles: continue
            if any(b in h for b in blocked_keywords): continue
            seen.add(vid)
            candidates.append((vid, h, it.get('category', 'general')))

    # 2. Load from candidates_500.json and candidates_200.json
    for jf in ['candidates_500.json', 'candidates_200.json', 'candidate_explore_reels.json']:
        if not os.path.exists(jf): continue
        with open(jf) as f:
            d = json.load(f)
        for it in d:
            v = it.get('video_id')
            h = (it.get('author') or '').lower()
            if v and v not in seen and h not in existing_handles and h not in user_rejected_handles:
                if any(b in h for b in blocked_keywords): continue
                seen.add(v)
                candidates.append((v, h, it.get('tag', 'general')))

    return candidates

def process_candidate(cand):
    vid, raw_h, cat = cand

    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    existing_handles = {v['handle'].lower() for v in manifest.values()}
    if raw_h in existing_handles or raw_h in user_rejected_handles:
        return None

    url = f"https://r.jina.ai/https://www.instagram.com/reel/{vid}/embed/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u:
            md = u.read().decode('utf-8')

        m_likes = re.search(r'\[([0-9,]+)\s+likes\]', md)
        if not m_likes: m_likes = re.search(r'([0-9,]+)\s+likes', md)
        if not m_likes: return None

        likes = int(m_likes.group(1).replace(',', ''))
        if likes < 50000: return None

        m_handle = re.search(r'\[([a-zA-Z0-9._-]+)\]\(https://www\.instagram\.com/[a-zA-Z0-9._-]+/\?utm_source=ig_embed', md)
        clean = m_handle.group(1) if m_handle else raw_h
        clean = re.sub(r'[^a-zA-Z0-9._-]', '', clean.split('\n')[0].strip())[:25]
        clean_lower = clean.lower()

        if clean_lower in existing_handles or clean_lower in user_rejected_handles:
            return None
        if any(b in clean_lower for b in blocked_keywords):
            return None

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
                ok, sc = loosened_eval(fr)
                if ok and sc > best_sc:
                    best_sc = sc
                    best_fr = fr.copy()
            except: pass

        if best_fr is None:
            return None

        # Thread-safe save
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

        print(f"  ✓ [{rank_str}/1000] [{cat}] SAVED: @{clean} -> {reel_url} ({likes:,} likes, conf: {best_sc:.1f}%)", flush=True)
        return clean

    except Exception:
        return None

def main():
    print("=== Launching Fast Stream Harvester with Loosened Face Evaluator ===")
    candidates = get_candidates()
    print(f"Total priority candidates queued: {len(candidates)}")

    for i, cand in enumerate(candidates):
        with open(VERIFIED_MANIFEST) as mf:
            m = json.load(mf)
        if len(m) >= 1000:
            print("=== Reached 1,000 Verified Faces! Complete! ===")
            break

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(candidates)}] Evaluated candidates... (Latest: @{cand[1]})", flush=True)

        process_candidate(cand)
        time.sleep(3.2)

if __name__ == "__main__":
    main()
