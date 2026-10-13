import asyncio
import os
import re
import cv2
import numpy as np
import requests
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

PURE_TAGS = [
    "publicspeaking", "tedtalk", "keynotespeaker", "speech", "communication",
    "bodylanguage", "careeradvice", "lifecoach", "psychologist", "stoicism"
]

os.makedirs("scratch/test_pure_th", exist_ok=True)

def score_talking_head_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: # MUST BE EXACTLY ONE FACE (SOLITARY PRESENTER!)
        return -1.0, None
        
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.85: return -1.0, None
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    # Must be talking-head framing: medium close-up or bust (3.5% to 22% of frame)
    if area_pct < 3.5 or area_pct > 22.0:
        return -1.0, None
        
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0: return -1.0, None
    
    # Looking directly at camera
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.28: return -1.0, None
    
    # Eyes must be above nose, nose above mouth
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, None
        
    # Check mouth separation
    mouth_w = abs(lm_x - rm_x)
    if mouth_w < eye_span * 0.40:
        return -1.0, None

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
        return -1.0, None
        
    # Sharpness
    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 35: return -1.0, None
    
    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40
    return score, face

async def test_run():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()
        
        # Scrape publicspeaking tag
        await page.goto('https://www.instagram.com/popular/publicspeaking/?utm_source=explore_tag', wait_until='domcontentloaded')
        await asyncio.sleep(0.5)
        cards = await page.evaluate(r'''() => {
            const anchors = document.querySelectorAll('a[href*=\"/reel/\"]');
            return Array.from(anchors).slice(0, 10).map(a => {
                const img = a.querySelector('img');
                const text = a.innerText || '';
                const parts = text.split('\\n');
                return {
                    href: a.href.split('?')[0],
                    author: (parts[0] || '').trim(),
                    views: (parts[1] || '').trim(),
                    caption: parts.slice(2).join(' ').trim()
                };
            });
        }''')
        
        saved_count = 0
        for c in cards:
            author = c['author']
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', c['href'])
            vid = m.group(1) if m else None
            if not vid: continue
            
            print(f"\nOpening {author} ({vid})...")
            embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
            try:
                await page.goto(embed_url, wait_until='domcontentloaded', timeout=6000)
                await asyncio.sleep(0.4)
                video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                if video_src:
                    cap = cv2.VideoCapture(video_src)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"  video_src found! total frames={total}")
                    step = max(1, total // 25)
                    best_f = None
                    best_s = -1.0
                    for idx in range(0, min(total, 450), step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret or frame is None: continue
                        s, _ = score_talking_head_frame(frame)
                        if s > best_s:
                            best_s = s
                            best_f = frame.copy()
                    cap.release()
                    
                    if best_f is not None:
                        out_p = f"scratch/test_pure_th/{author}_{vid}.jpg"
                        cv2.imwrite(out_p, best_f)
                        print(f"  ✓ SAVED {out_p}: shape={best_f.shape}, score={best_s:.1f}")
                        saved_count += 1
                    else:
                        print(f"  ✗ No frame satisfied strict solitary talking-head criteria")
                else:
                    print(f"  ✗ No video_src")
            except Exception as e:
                print(f"  Error: {e}")
                
        await ctx.close()
        print(f"\nSaved {saved_count} pure talking-head screenshots!")

asyncio.run(test_run())
