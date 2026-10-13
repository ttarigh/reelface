import asyncio
import os
import json
import re
import cv2
import numpy as np
import urllib.request
import ssl
from playwright.async_api import async_playwright

VERIFIED_MANIFEST = 'assets/verified_50k_manifest.json'
CREATORS_JSON = 'assets/creators.json'
OUTPUT_DIR = 'talking_head_photos_100'
BLACKLIST_PATH = 'assets/rejected_blacklist.json'
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

def ear(pts):
    p2_p6 = np.linalg.norm(pts[37] - pts[41])
    p3_p5 = np.linalg.norm(pts[38] - pts[40])
    p1_p4 = np.linalg.norm(pts[36] - pts[39])
    left_ear = (p2_p6 + p3_p5) / (2.0 * p1_p4 + 1e-5)

    p2_p6_r = np.linalg.norm(pts[43] - pts[47])
    p3_p5_r = np.linalg.norm(pts[44] - pts[46])
    p1_p4_r = np.linalg.norm(pts[42] - pts[45])
    right_ear = (p2_p6_r + p3_p5_r) / (2.0 * p1_p4_r + 1e-5)

    return (left_ear + right_ear) / 2.0

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0

    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    # Strictly single face in frame (no two-person scenes, skits, street interviews)
    if faces is None or len(faces) != 1: return False, 0

    primary = faces[0]
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

async def process_candidate(page, cand, detector, facemark, lock):
    vid = cand.get('video_id')
    tag = cand.get('tag', 'general')

    blocked_keywords = [
        'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
        'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
        'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
        'derm', 'dermatolog', 'dr.', 'dr_', 'dr', 'accent', 'pronounc', 'esl', 'medical', 'md', 'pharma',
        'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german', 'japanese', 'chinese',
        'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language', 'podcast', 'news', 'music', 'singer',
        'radio', 'band', 'guitar', 'vogue', 'hollywood', 'unlocked'
    ]

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
            const img = document.querySelector('.EmbeddedMediaImage');
            return { likes, handle, img_src: img ? img.src : null };
        }''')

        likes = data.get('likes', 0)
        if likes < 25000:
            return False

        raw_handle = data.get('handle') or f"creator_{vid[:8]}"
        clean = re.sub(r'[^a-zA-Z0-9._-]', '', raw_handle.split('\n')[0].strip())[:25]
        clean_lower = clean.lower()

        if any(b in clean_lower for b in blocked_keywords):
            return False

        with open(VERIFIED_MANIFEST) as mf:
            current_manifest = json.load(mf)
        existing_handles = {v['handle'].lower() for v in current_manifest.values()}
        if clean_lower in existing_handles:
            return False

        img_src = data.get('img_src')
        if not img_src:
            return False

        req = urllib.request.Request(img_src, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5.0, context=ctx_ssl) as u:
            arr = np.frombuffer(u.read(), dtype=np.uint8)
            fr = cv2.imdecode(arr, -1)

        if fr is None:
            return False

        ok, score = eval_frame(fr, detector, facemark)
        if not ok:
            return False

        async with lock:
            with open(VERIFIED_MANIFEST) as mf:
                manifest = json.load(mf)

            existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
            next_rank = max(existing_ranks + [0]) + 1
            rank_str = f"{next_rank:04d}"
            out_name = f"{rank_str}_{clean}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, fr)

            reel_url = f"https://www.instagram.com/reel/{vid}/"
            manifest[rank_str] = {
                'rank': rank_str,
                'handle': clean,
                'video_id': vid,
                'reelUrl': reel_url,
                'likes': likes,
                'photo': out_name,
                'category': tag,
                'score': round(score, 1),
                'verified': True,
                'source': f'popular_{tag}'
            }
            with open(VERIFIED_MANIFEST, 'w') as mf:
                json.dump(manifest, mf, indent=2)

            with open(CREATORS_JSON, 'w') as cf:
                json.dump(list(manifest.values()), cf, indent=2)

            print(f"  ✓ [{rank_str}] [{tag}] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {score:.1f})", flush=True)
            return True

    except Exception as e:
        return False

async def worker(worker_id, queue, lock, browser):
    detector, facemark = create_models()
    context = await browser.new_context(viewport={'width': 600, 'height': 900})
    page = await context.new_page()

    count = 0
    while True:
        try:
            cand = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        count += 1
        if count % 10 == 0:
            print(f"  [W{worker_id}] Processed {count} items... (queue remaining: {queue.qsize()})", flush=True)

        await process_candidate(page, cand, detector, facemark, lock)

    await context.close()

async def main():
    print("=== Launching Popular Stream Harvester ===")
    lock = asyncio.Lock()

    with open('fresh_popular_candidates.json') as f:
        candidates = json.load(f)

    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    seen_vids = set(v.get('video_id') for v in manifest.values() if v.get('video_id'))

    queue = asyncio.Queue()
    for c in candidates:
        if c.get('video_id') and c['video_id'] not in seen_vids:
            queue.put_nowait(c)

    print(f"Queued {queue.qsize()} fresh candidates from popular explore tags.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        workers = [worker(i, queue, lock, browser) for i in range(5)]
        await asyncio.gather(*workers)
        await browser.close()

    print("=== Harvester Complete ===")

if __name__ == '__main__':
    asyncio.run(main())
