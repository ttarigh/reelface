import asyncio
import os
import re
import json
import cv2
import numpy as np
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

LBF_MODEL = "models/lbfmodel.yaml"
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL)

OUTPUT_DIR = "talking_head_photos_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MANIFEST_PATH = "talking_heads_100_manifest.json"
CANDIDATES_PATH = "all_pro_candidates.json"

def norm_handle(h):
    if not h: return ""
    clean = re.sub(r'[^a-zA-Z0-9]', '', h.lower())
    return clean[:6]

def ear(pts):
    # Right eye EAR
    r_h1 = np.linalg.norm(pts[37] - pts[41])
    r_h2 = np.linalg.norm(pts[38] - pts[40])
    r_w  = np.linalg.norm(pts[36] - pts[39])
    r_ear = (r_h1 + r_h2) / (2.0 * r_w + 1e-5)
    
    # Left eye EAR
    l_h1 = np.linalg.norm(pts[43] - pts[47])
    l_h2 = np.linalg.norm(pts[44] - pts[46])
    l_w  = np.linalg.norm(pts[42] - pts[45])
    l_ear = (l_h1 + l_h2) / (2.0 * l_w + 1e-5)
    
    return (r_ear + l_ear) / 2.0

def eval_frame(fr):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    top = fr[:int(h*0.06), :]
    bot = fr[int(h*0.94):, :]
    if (np.mean(top) < 15 and np.std(top) < 10) or (np.mean(bot) < 15 and np.std(bot) < 10):
        return False, 0
        
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return False, 0
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area = (fw*fh)/(w*h)
    if area < 0.040 or area > 0.17: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.17: return False, 0
    if fy < int(h*0.10) or fy > int(h*0.50): return False, 0
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    eye_span = abs(le_x - re_x)
    if eye_span <= 14: return False, 0
    if abs((nt_x - re_x) - (le_x - nt_x)) / eye_span > 0.18: return False, 0
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    # Eye openness / gaze
    eye_ratio = ear(pts)
    if eye_ratio < 0.26: return False, 0
    
    # Mouth open speaking
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.28 or m_ratio > 0.85: return False, 0
    
    # Neck / chin area check
    chin_y = int(pts[8, 1])
    chin_x = int(pts[8, 0])
    neck_h = min(h - chin_y, 70)
    if neck_h > 20:
        neck = gray[chin_y:chin_y+neck_h, max(0, chin_x-40):min(w, chin_x+40)]
        if np.max(np.abs(cv2.Sobel(neck, cv2.CV_64F, 0, 1, ksize=3))) > 1050:
            return False, 0
            
    score = conf*100 + m_ratio*50 + eye_ratio*50
    return True, score

async def harvest_100():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)

    existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith('composited') and not f.startswith('procrustes')])
    current_count = len(existing_files)
    print(f"Starting harvest from count: {current_count}/100")

    seen_prefixes = set()
    for m in manifest:
        p = norm_handle(m['author'])
        if p: seen_prefixes.add(p)

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for cand in candidates:
            if current_count >= 100:
                print("\nReached target of 100 talking head faces!")
                break

            author = cand.get('author', '').split('\n')[0].strip().lstrip('@')
            clean_author = re.sub(r'[^a-zA-Z0-9._-]', '', author)[:25]
            prefix = norm_handle(author)
            if not prefix or prefix in seen_prefixes:
                continue

            vid = cand.get('video_id')
            if not vid: continue

            try:
                await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=4000)
                el = await page.wait_for_selector('video', timeout=2500)
                v_src = await el.get_attribute('src')
                if not v_src: continue

                cap = cv2.VideoCapture(v_src)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 35)
                best_f = None
                best_s = -1.0

                for idx in range(10, min(total, 450), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None: continue
                    ok, sc = eval_frame(frame)
                    if ok and sc > best_s:
                        best_s = sc
                        best_f = frame.copy()
                cap.release()

                if best_f is not None and best_s > 0:
                    current_count += 1
                    out_name = f"{current_count:03d}_{clean_author}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    cv2.imwrite(out_path, best_f)
                    seen_prefixes.add(prefix)
                    manifest.append({
                        'rank': current_count,
                        'author': clean_author,
                        'video_id': vid,
                        'video_url': cand.get('href'),
                        'image_path': out_path,
                        'score': best_s
                    })
                    with open(MANIFEST_PATH, 'w') as mf:
                        json.dump(manifest, mf, indent=2)
                    print(f"  ✓ [{current_count:03d}/100] SAVED (#{cand.get('tag')}): @{clean_author} -> {out_name} (score: {best_s:.1f})")
            except Exception:
                pass

        await ctx.close()

    print(f"\nHarvest finished with {current_count}/100 talking head faces.")

if __name__ == "__main__":
    asyncio.run(harvest_100())
