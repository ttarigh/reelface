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
CANDIDATES_PATH = "candidate_explore_reels.json"

PRO_TAGS = {
    'dermatologist', 'psychologist', 'doctoradvice', 'neuroscience',
    'careeradvice', 'interviewtips', 'tedtalks', 'communicationskills',
    'executivepresence', 'publicspeaking', 'speechcoach'
}

def norm_handle(h):
    if not h: return ""
    clean = re.sub(r'[^a-zA-Z0-9]', '', h.lower())
    return clean[:6]

def score_talking_head(frame):
    h, w = frame.shape[:2]
    if h <= w: return -1.0, "Landscape"
    
    # 1. Reject letterboxing (black bars)
    top_bar = frame[:int(h*0.06), :]
    bot_bar = frame[int(h*0.94):, :]
    if (np.mean(top_bar) < 15 and np.std(top_bar) < 10) or (np.mean(bot_bar) < 15 and np.std(bot_bar) < 10):
        return -1.0, "Letterbox black bars"

    # 2. Reject horizontal split-screen seam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mid_strip = gray[int(h*0.42):int(h*0.58), :]
    grad_y = cv2.Sobel(mid_strip, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.mean(np.abs(grad_y), axis=1)) > 75:
        return -1.0, "Split screen seam detected"

    # 3. Single face detection
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: return -1.0, "Not 1 face"
    
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.90: return -1.0, "Low conf"
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    if area_pct < 4.5 or area_pct > 17.0: return -1.0, f"Area {area_pct:.1f}%"
    
    # Must be horizontally centered (bust shot)
    center_x = fx + fw / 2.0
    if abs(center_x - w / 2.0) > (w * 0.18):
        return -1.0, "Off-center face"
    
    # Upper half of vertical frame
    if fy < int(h * 0.12) or fy > int(h * 0.48):
        return -1.0, f"Face vertical pos fy={fy}"
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 14: return -1.0, "Eye span"
    
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.18: return -1.0, f"Looking away ({nose_offset:.2f})"
    
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, "Landmark order"

    # 4. Check for occlusions in chin / lower face
    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, "Empty lower face"
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel_h)) > 1300: return -1.0, "Occlusion horizontal edge"
    
    sobel_v = cv2.Sobel(lower_gray, cv2.CV_64F, 1, 0, ksize=3)
    if np.max(np.abs(sobel_v)) > 1300: return -1.0, "Occlusion vertical edge"

    # 5. Facemark 68 landmarks validation
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0:
        return -1.0, "Facemark 68 fit failed"
    pts = lms[0].reshape(68, 2)
    
    # Mouth aspect ratio: actively speaking
    mouth_w = np.linalg.norm(pts[48] - pts[54])
    mouth_h = np.linalg.norm(pts[51] - pts[57])
    mouth_ratio = mouth_h / (mouth_w + 1e-5)
    if mouth_ratio < 0.22 or mouth_ratio > 0.85:
        return -1.0, f"Mouth closed ({mouth_ratio:.2f})"

    # Eye openness
    eye_w, eye_h = max(4, int(fw * 0.16)), max(4, int(fh * 0.12))
    def eye_stats(ex, ey):
        x1, x2 = max(0, ex - eye_w//2), min(w, ex + eye_w//2)
        y1, y2 = max(0, ey - eye_h//2), min(h, ey + eye_h//2)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0: return 0, 0
        min_v, max_v, _, _ = cv2.minMaxLoc(patch)
        sobel = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        return (max_v - min_v), np.mean(np.abs(sobel))
        
    r_con, r_edge = eye_stats(re_x, re_y)
    l_con, l_edge = eye_stats(le_x, le_y)
    if r_con < 42 or l_con < 42 or r_edge < 14 or l_edge < 14:
        return -1.0, "Eyes not wide open"

    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 45: return -1.0, "Blurry"

    score = conf * 100 + min(r_edge, l_edge)*2.0 + min(r_con, l_con)*0.4 + min(50, sharpness*0.05) - nose_offset*50 + mouth_ratio*50
    return score, "OK"

async def harvest_pro():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)

    pro_candidates = [c for c in candidates if c.get('tag') in PRO_TAGS]
    print(f"Loaded {len(pro_candidates)} professional candidates.")

    existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith('composited') and not f.startswith('procrustes')])
    current_count = len(existing_files)
    print(f"Current verified count: {current_count}/100")

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

        for cand in pro_candidates:
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
                await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=3500)
                await asyncio.sleep(0.2)
                v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                if not v_src: continue

                cap = cv2.VideoCapture(v_src)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 35)
                best_f = None
                best_s = -1.0

                for idx in range(0, min(total, 500), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None: continue
                    s, msg = score_talking_head(frame)
                    if s > best_s:
                        best_s = s
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

    print(f"\nFinal count reached: {current_count}/100")

if __name__ == "__main__":
    asyncio.run(harvest_pro())
