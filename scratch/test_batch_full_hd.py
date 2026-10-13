import asyncio
import cv2
import numpy as np
import os
import sqlite3
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()
c.execute("SELECT id, creator_name, video_id, video_url, like_count FROM videos ORDER BY like_count DESC LIMIT 10")
test_rows = c.fetchall()

os.makedirs("scratch/test_batch_full_hd", exist_ok=True)

def find_best_frame_from_video(video_src, max_samples=25):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        return None, -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_samples) if total > 0 else 20
    
    best_frame = None
    best_score = -1.0
    
    for idx in range(0, min(total, 450), step):
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
        if nose_offset > 0.38: continue # looking towards camera
        
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
        if r_con < 40 or l_con < 40 or r_edge < 12 or l_edge < 12:
            continue # eyes must be open
            
        # Sharpness
        fx, fy, fw, fh = [int(v) for v in face[:4]]
        face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
        if sharpness < 40: continue
        
        score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            
    cap.release()
    return best_frame, best_score

async def run_batch():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()
        
        for rank, (vid, name, v_id, v_url, likes) in enumerate(test_rows, 1):
            embed_url = f"https://www.instagram.com/reel/{v_id}/embed/"
            print(f"[{rank}/10] Loading {name} ({embed_url})...")
            try:
                await page.goto(embed_url, wait_until='domcontentloaded', timeout=8000)
                await asyncio.sleep(0.6)
                video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                if not video_src:
                    # try clicking play button
                    await page.evaluate("() => document.querySelector('.EmbeddedMedia, a[href*=\"/reel/\"]')?.click()")
                    await asyncio.sleep(0.8)
                    video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                
                if video_src:
                    frame, score = find_best_frame_from_video(video_src)
                    if frame is not None:
                        out_file = f"scratch/test_batch_full_hd/{rank:03d}_{name.lstrip('@')}.jpg"
                        cv2.imwrite(out_file, frame)
                        print(f"  ✓ SAVED direct video screenshot {out_file}: shape {frame.shape}, score {score:.1f}")
                    else:
                        print(f"  ✗ No valid frame met criteria for {name}")
                else:
                    print(f"  ✗ No video src found for {name}")
            except Exception as e:
                print(f"  Error on {name}: {e}")
                
        await ctx.close()

asyncio.run(run_batch())
