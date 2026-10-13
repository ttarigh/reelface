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

TAGS = [
    "publicspeaking", "communicationskills", "executivepresence", "relationshiptips",
    "lifecoach", "dailywisdom", "dermatologist", "psychologist", "mentalhealthtips",
    "confidencecoach", "careeradvice", "leadershipskills", "neuroscience", "tedtalks",
    "doctoradvice", "stoicism", "voicetrainer", "speechcoach", "communicationtips"
]

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
    mid_strip = gray[int(h*0.45):int(h*0.55), :]
    grad_y = cv2.Sobel(mid_strip, cv2.CV_64F, 0, 1, ksize=3)
    row_means = np.mean(np.abs(grad_y), axis=1)
    if np.max(row_means) > 85:
        return -1.0, "Split screen seam detected"

    # 3. Single face detection
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: return -1.0, "Not 1 face"
    
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return -1.0, "Low conf"
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    if area_pct < 4.0 or area_pct > 18.0: return -1.0, f"Area {area_pct:.1f}%"
    
    # Face position: should be centered in upper-middle half
    if fy < int(h * 0.10) or fy > int(h * 0.55):
        return -1.0, f"Face position fy={fy}"
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 12: return -1.0, "Eye span"
    
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.20: return -1.0, f"Looking away ({nose_offset:.2f})"
    
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, "Landmark order"

    # 4. Check for occlusions in lower face (hands, books, objects)
    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, "Empty lower face"
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel_h)) > 1500: return -1.0, "Occlusion horizontal edge"
    
    sobel_v = cv2.Sobel(lower_gray, cv2.CV_64F, 1, 0, ksize=3)
    if np.max(np.abs(sobel_v)) > 1500: return -1.0, "Occlusion vertical edge"

    # 5. Facemark 68 landmarks validation
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0:
        return -1.0, "Facemark 68 fit failed"
    pts = lms[0].reshape(68, 2)
    
    # Mouth aspect ratio: actively speaking
    mouth_w = np.linalg.norm(pts[48] - pts[54])
    mouth_h = np.linalg.norm(pts[51] - pts[57])
    mouth_ratio = mouth_h / (mouth_w + 1e-5)
    if mouth_ratio < 0.22:
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
    if r_con < 40 or l_con < 40 or r_edge < 13 or l_edge < 13:
        return -1.0, "Eyes not wide open"

    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 40: return -1.0, "Blurry"

    score = conf * 100 + min(r_edge, l_edge)*2.0 + min(r_con, l_con)*0.4 + min(50, sharpness*0.05) - nose_offset*50 + mouth_ratio*50
    return score, "OK"

def clean_handle(h):
    if not h: return ""
    h = h.split('\n')[0].strip().lstrip('@')
    return re.sub(r'[^a-zA-Z0-9._-]', '', h)[:30]

async def harvest_pure_talking_heads():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith('composited') and not f.startswith('procrustes')])
    current_count = len(existing_files)
    print(f"Current count in {OUTPUT_DIR}: {current_count}/100")

    existing_handles = set([m['author'].lower() for m in manifest])

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for tag in TAGS:
            if current_count >= 100: break
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            print(f"\n>>> Browsing #{tag} (need {100 - current_count} more)...")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=6000)
                await asyncio.sleep(0.3)
                
                # Scroll down twice to load 24-36 reels
                for _ in range(2):
                    await page.evaluate("window.scrollBy(0, 1500)")
                    await asyncio.sleep(0.4)

                reels = await page.evaluate(r"""() => {
                    const anchors = document.querySelectorAll('a[href*="/reel/"]');
                    return Array.from(anchors).map(a => ({
                        href: a.href.split('?')[0],
                        author: (a.innerText || '').split('\n')[0].trim()
                    }));
                }""")

                print(f"  Found {len(reels)} reels on #{tag}")

                for r in reels:
                    if current_count >= 100: break
                    h = clean_handle(r['author'])
                    if not h or h.lower() in existing_handles: continue
                    m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                    vid = m.group(1) if m else None
                    if not vid: continue

                    try:
                        await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=4000)
                        await asyncio.sleep(0.25)
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
                            out_name = f"{current_count:03d}_{h}.jpg"
                            out_path = os.path.join(OUTPUT_DIR, out_name)
                            cv2.imwrite(out_path, best_f)
                            existing_handles.add(h.lower())
                            manifest.append({
                                'rank': current_count,
                                'author': h,
                                'video_id': vid,
                                'video_url': r['href'],
                                'image_path': out_path,
                                'score': best_s
                            })
                            with open(MANIFEST_PATH, 'w') as mf:
                                json.dump(manifest, mf, indent=2)
                            print(f"  ✓ [{current_count:03d}/100] SAVED: @{h} -> {out_name} ({best_f.shape}, score: {best_s:.1f})")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error on #{tag}: {e}")

        await ctx.close()

    print(f"\nFinal count in {OUTPUT_DIR}: {current_count}/100")

if __name__ == "__main__":
    asyncio.run(harvest_pure_talking_heads())
