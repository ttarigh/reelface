import asyncio
import os
import re
import sqlite3
import cv2
import numpy as np
import requests
import time
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

def check_play_button(img):
    if img is None or img.size == 0: return False
    h, w = img.shape[:2]
    ch, cw = h // 2, w // 2
    rh, rw = int(h * 0.30), int(w * 0.30)
    roi = img[max(0, ch - rh//2):min(h, ch + rh//2), max(0, cw - rw//2):min(w, cw + rw//2)]
    if roi.size == 0: return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white_mask = (gray > 238).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
            if len(approx) == 3 or (len(approx) <= 5 and area > 200):
                return True
    return False

def extract_best_video_screenshot(video_src, max_samples=20):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_samples) if total > 0 else 20
    
    best_frame = None
    best_score = -1.0
    
    for idx in range(0, min(total, 400), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0: continue
        
        face = max(faces, key=lambda f: f[2]*f[3])
        conf = float(face[-1])
        if conf < 0.85: continue
        
        re_x, re_y = int(face[4]), int(face[5])
        le_x, le_y = int(face[6]), int(face[7])
        nt_x, nt_y = int(face[8]), int(face[9])
        
        eye_span = abs(le_x - re_x)
        if eye_span <= 0: continue
        nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
        if nose_offset > 0.38: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_w, eye_h = max(4, int(face[2]*0.16)), max(4, int(face[3]*0.12))
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
        if r_con < 38 or l_con < 38 or r_edge < 11 or l_edge < 11:
            continue
            
        # Sharpness
        fx, fy, fw, fh = [int(v) for v in face[:4]]
        face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
        if sharpness < 35: continue
        
        score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            
    cap.release()
    return best_frame, best_score

async def test_speed():
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT id, creator_name, video_id, video_url FROM videos ORDER BY like_count DESC LIMIT 15")
    rows = c.fetchall()
    conn.close()

    t0 = time.time()
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        async def fetch_one(vid, name, v_id):
            page = await ctx.new_page()
            url = f"https://www.instagram.com/reel/{v_id}/embed/"
            v_src = None
            i_src = None
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=6000)
                await asyncio.sleep(0.4)
                v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                i_src = await page.evaluate("() => document.querySelector('.EmbeddedMediaImage')?.src || null")
            except Exception:
                pass
            finally:
                await page.close()
            return name, v_id, v_src, i_src

        # Batch of 6 concurrent tabs
        tasks = [fetch_one(r[0], r[1], r[2]) for r in rows]
        results = await asyncio.gather(*tasks)
        await ctx.close()

    dt = time.time() - t0
    print(f"Fetched 15 embeds concurrently in {dt:.1f}s! ({dt/15:.2f}s per reel)")
    
    # Process frames
    for name, v_id, v_src, i_src in results:
        if v_src:
            frame, score = extract_best_video_screenshot(v_src)
            has_pb = check_play_button(frame) if frame is not None else False
            print(f"  {name:22}: Video Screenshot shape={frame.shape if frame is not None else None}, Score={score:.1f}, PlayButton={has_pb}")
        elif i_src:
            resp = requests.get(i_src, timeout=4)
            nparr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            has_pb = check_play_button(img) if img is not None else False
            print(f"  {name:22}: Embed Image shape={img.shape if img is not None else None}, PlayButton={has_pb}")

asyncio.run(test_speed())
