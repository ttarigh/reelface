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
BLACKLIST_PATH = "assets/rejected_blacklist.json"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

ctx_ssl = ssl._create_unverified_context()
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.5)

blocked_keywords = [
    # Medical & Dental
    'doctor', 'derm', 'clinic', 'dentist', 'surgery', 'dr.', 'dr_', 'dr', 'hospital', 'nurse',
    'scrubs', 'healthcare', 'pharma', 'physio', 'chiro', 'dental', 'orthodont', 'teeth',
    # Podcasts & Interviews
    'podcast', 'pod', 'clip', 'episode', 'ep.', 'interview', 'guest', 'host', 'show',
    'channel', 'radio', 'studio', 'broadcast', 'anchor',
    # Stand-up Comedy Stages
    'standup', 'comic', 'comedyclub', 'stage', 'theatre',
    # Language Tutors & Pronunciation
    'english', 'spanish', 'french', 'german', 'japanese', 'chinese', 'korean', 'italian',
    'russian', 'accent', 'pronounc', 'tutor', 'grammar', 'vocab', 'learn', 'lingo', 'polyglot',
    # Life Coaches & Gimmicks
    'coach', 'guru', 'mindset', 'palmreader', 'astrology', 'tarot', 'spiritual', 'manifest',
    'crypto', 'forex', 'dropship', 'affiliate', 'realestate', 'wealth'
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
    # Head & shoulders framing (at least 3.5% of frame)
    if area < 0.035: return False, 0
    if conf < 0.68: return False, 0
    return True, conf * 100

def get_clean_candidates():
    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    existing_handles = {v['handle'].lower() for v in manifest.values()}
    existing_vids = {v.get('video_id') for v in manifest.values() if v.get('video_id')}

    bl_handles = set()
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH) as bf:
            bl_data = json.load(bf)
            bl_handles = {h.lower() for h in bl_data.get("handles", [])}

    candidates = []
    seen = set(existing_vids)

    # Load from candidate_explore_reels.json, candidates_500.json, candidates_200.json
    source_files = [
        'candidate_explore_reels.json',
        'candidates_500.json',
        'candidates_200.json',
        'assets/recovered_reels_manifest.json'
    ]

    for jf in source_files:
        if not os.path.exists(jf): continue
        with open(jf) as f:
            d = json.load(f)
        items = list(d.values()) if isinstance(d, dict) else d
        for it in items:
            h = (it.get('author') or it.get('handle') or '').lower()
            tag = (it.get('tag') or it.get('category') or '').lower()
            url = it.get('href') or it.get('reelUrl') or ''
            v = it.get('video_id')
            if not v and url:
                m = re.search(r'/reel/([a-zA-Z0-9_-]+)', url)
                if m: v = m.group(1)

            if not v or not h or v in seen: continue
            seen.add(v)
            if h in existing_handles or h in bl_handles: continue
            if any(b in h for b in blocked_keywords) or any(b in tag for b in blocked_keywords): continue

            # Prioritize storytime, pov, music, lifestyle, booktok, creative
            priority = 0
            if any(t in tag for t in ['storytime', 'pov', 'music', 'lifestyle', 'grwm', 'book', 'creative', 'voice']):
                priority = 1

            candidates.append((v, h, tag, priority))

    # Sort priority candidates first
    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates

def check_and_save(cand):
    v, raw_h, tag, prio = cand

    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    existing_handles = {v['handle'].lower() for v in manifest.values()}
    if raw_h in existing_handles: return False

    time.sleep(1.2)
    url = f"https://r.jina.ai/https://www.instagram.com/reel/{v}/embed/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=8.0, context=ctx_ssl) as u:
            md = u.read().decode('utf-8')

        m_likes = re.search(r'\[([0-9,]+)\s+likes\]', md)
        if not m_likes: m_likes = re.search(r'([0-9,]+)\s+likes', md)
        if not m_likes: return False

        likes = int(m_likes.group(1).replace(',', ''))
        if likes < 50000: return False

        m_handle = re.search(r'\[([a-zA-Z0-9._-]+)\]\(https://www\.instagram\.com/[a-zA-Z0-9._-]+/\?utm_source=ig_embed', md)
        clean = m_handle.group(1) if m_handle else raw_h
        clean = re.sub(r'[^a-zA-Z0-9._-]', '', clean.split('\n')[0].strip())[:25]
        clean_lower = clean.lower()

        if clean_lower in existing_handles: return False
        if any(b in clean_lower for b in blocked_keywords): return False

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

        if best_fr is None: return False

        # Save to database
        with open(VERIFIED_MANIFEST) as mf:
            manifest = json.load(mf)

        existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
        next_rank = max(existing_ranks + [0]) + 1
        rank_str = f"{next_rank:04d}"
        out_name = f"{rank_str}_{clean}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, best_fr)

        reel_url = f"https://www.instagram.com/reel/{v}/"
        cat = tag if tag else "lifestyle_storytime"
        manifest[rank_str] = {
            'rank': rank_str,
            'handle': clean,
            'video_id': v,
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
        return True

    except Exception:
        return False

def main():
    print("=== Launching Targeted Viral Talking-Head Harvester ===")
    candidates = get_clean_candidates()
    print(f"Total clean, prioritized candidates queued: {len(candidates)}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        for i, cand in enumerate(candidates):
            with open(VERIFIED_MANIFEST) as mf:
                manifest = json.load(mf)
            if len(manifest) >= 1000:
                print("=== Reached 1,000 Verified Faces! Complete! ===")
                break

            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(candidates)}] Evaluated candidates... (Latest: @{cand[1]})", flush=True)

            executor.submit(check_and_save, cand)
            time.sleep(2.5)

if __name__ == "__main__":
    main()
