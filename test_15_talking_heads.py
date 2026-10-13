import asyncio
import os
import re
import cv2
import numpy as np
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

# Test candidate list of verified talking head reels
CANDIDATES = [
    ("Maria Lamia", "marialamiabou", "DbGpEhssxxr", "https://www.instagram.com/reel/DbGpEhssxxr/"),
    ("Austin Nasso", "austinnasso", "DVeOL-CDyhe", "https://www.instagram.com/reel/DVeOL-CDyhe/"),
    ("Joe Navarro", "mind_insightz", "DanQLAWxxx6", "https://www.instagram.com/reel/DanQLAWxxx6/"),
    ("Mel Robbins", "melrobbins", "DNRLiUhixek", "https://www.instagram.com/reel/DNRLiUhixek/"),
    ("Dr. Mike", "doctor.mike", "DNVwq35RrJc", "https://www.instagram.com/reel/DNVwq35RrJc/"),
    ("Jazz Thornton", "jazzthornton", "C6fNkCsPF9e", "https://www.instagram.com/reel/C6fNkCsPF9e/"),
    ("Giancarlo Esposito", "giancarloesposito", "DVtnI2vESMg", "https://www.instagram.com/reel/DVtnI2vESMg/"),
    ("Kyle Nutt", "kylenutt", "C0t4ot2uPWd", "https://www.instagram.com/reel/C0t4ot2uPWd/"),
    ("David Goggins", "davidgoggins", "DOPJ99Ggc9I", "https://www.instagram.com/reel/DOPJ99Ggc9I/"),
    ("Katherine Choi", "katherinechoiiii", "DTLMHzXDm2w", "https://www.instagram.com/reel/DTLMHzXDm2w/"),
    ("Enora Hope", "enorahope", "DSW87rWjIn0", "https://www.instagram.com/reel/DSW87rWjIn0/"),
    ("Julien Himself", "julienhimself", "C4zQ7fLp1Nq", "https://www.instagram.com/reel/C4zQ7fLp1Nq/"),
    ("Cheryl Porter", "cherylporterdiva", "C8qX-oNuu5y", "https://www.instagram.com/reel/C8qX-oNuu5y/"),
    ("Dr. Dray", "drdrayzday", "DBdpYJ8pVeq", "https://www.instagram.com/reel/DBdpYJ8pVeq/"),
    ("Paul McKenna", "iampaulmckenna", "C6mP2xIovWz", "https://www.instagram.com/reel/C6mP2xIovWz/"),
    ("Dan Brule", "danbruleofficial", "C7d8y91uM_Z", "https://www.instagram.com/reel/C7d8y91uM_Z/"),
    ("Rachel Rodgers", "rachrodgersesq", "C7-P_x2uV_r", "https://www.instagram.com/reel/C7-P_x2uV_r/"),
    ("Jay Lee Draney", "jayleedraney", "DVIUJgHjc0J", "https://www.instagram.com/reel/DVIUJgHjc0J/")
]

os.makedirs("scratch/test_15_heads", exist_ok=True)

def score_talking_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: return -1.0, "Not 1 face"
    
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.85: return -1.0, "Low conf"
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    # Must be classic bust/talking head shot (3.5% to 22%)
    if area_pct < 3.5 or area_pct > 22.0: return -1.0, f"Area {area_pct:.1f}%"
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0: return -1.0, "Eye span"
    
    # Looking directly at camera
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.28: return -1.0, f"Looking away ({nose_offset:.2f})"
    
    # Eyes above nose, nose above mouth
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, "Landmark order"
        
    # Check mouth:
    mouth_w = abs(lm_x - rm_x)
    if mouth_w < eye_span * 0.40: return -1.0, "Mouth narrow"

    # Lower face obstruction check:
    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, "Empty lower face"
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel_h)) > 1900: return -1.0, "Occlusion edge"

    # Open mouth check (actively talking):
    # In mouth patch, vertical gradient & intensity variance:
    my = (rm_y + lm_y) // 2
    mouth_patch = lower_gray[max(0, my - nt_y - 10):min(lower_gray.shape[0], my - nt_y + 15), :]
    mouth_var = np.std(mouth_patch) if mouth_patch.size > 0 else 0
    # Talking mouths have teeth/oral contrast (variance >= 18)
    
    # Eye openness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    if r_con < 38 or l_con < 38 or r_edge < 12 or l_edge < 12:
        return -1.0, "Eyes squinting/closed"

    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 35: return -1.0, "Blurry"

    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40 + min(30, mouth_var*0.5)
    return score, "OK"

async def test_all():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for name, handle, vid, url in CANDIDATES:
            print(f"\nProcessing {name} ({vid})...")
            try:
                await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=6500)
                await asyncio.sleep(0.4)
                v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                if not v_src:
                    print(f"  ✗ No video_src")
                    continue

                cap = cv2.VideoCapture(v_src)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 30)
                best_f = None
                best_s = -1.0

                for idx in range(0, min(total, 450), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None: continue
                    s, msg = score_talking_frame(frame)
                    if s > best_s:
                        best_s = s
                        best_f = frame.copy()
                cap.release()

                if best_f is not None:
                    out_p = f"scratch/test_15_heads/{handle}.jpg"
                    cv2.imwrite(out_p, best_f)
                    print(f"  ✓ SAVED {out_p}: {best_f.shape}, score: {best_s:.1f}")
                else:
                    print(f"  ✗ No frame passed strict talking head validation")
            except Exception as e:
                print(f"  Error: {e}")

        await ctx.close()

asyncio.run(test_all())
