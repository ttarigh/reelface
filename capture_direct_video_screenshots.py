"""
Direct Video Screenshot Capture Agent
Captures clear, full-resolution direct video screenshots (720x1280+ HD)
of talking-head creators unobstructed with their eyes wide open,
completely free of any play button overlays.
"""

import asyncio
import os
import re
import sqlite3
import cv2
import numpy as np
import requests
import time
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

def clean_handle_name(name):
    clean = re.sub(r'[^a-zA-Z0-9._-]', '', name.lstrip('@'))
    return clean[:30]

def extract_clean_video_id(v_url, v_id):
    m = re.search(r'/(?:reel|p)/([a-zA-Z0-9_-]+)', v_url)
    if m:
        return m.group(1)
    m2 = re.search(r'([a-zA-Z0-9_-]{9,})', v_id)
    if m2:
        return m2.group(1)
    return v_id

def analyze_and_score_frame(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return -1.0, None
        
    face = max(faces, key=lambda f: f[2]*f[3])
    conf = float(face[-1])
    if conf < 0.80:
        return -1.0, None
        
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    if fw < 40 or fh < 40:
        return -1.0, None

    # Landmarks: re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0:
        return -1.0, None
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.45: # looking towards camera
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
    
    # Eyes must be open!
    if r_con < 25 or l_con < 25:
        return -1.0, None

    # Sharpness
    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 20:
        return -1.0, None

    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*30
    return score, face

def get_best_frame_from_video(video_src, max_samples=25):
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        return None
        
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_samples) if total > 0 else 15
    
    best_frame = None
    best_score = -1.0
    
    for idx in range(0, min(total, 450), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        score, _ = analyze_and_score_frame(detector, frame)
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            
    cap.release()
    return best_frame

def get_best_frame_from_image(img_bytes):
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return None
    
    score, _ = analyze_and_score_frame(detector, frame)
    if score > 0:
        return frame
    return None

async def run_capture(batch_size=8):
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT id, creator_name, video_id, video_url, thumbnail_url, like_count FROM videos ORDER BY like_count DESC")
    videos = c.fetchall()
    conn.close()

    print(f"Loaded {len(videos)} videos from viral_videos.db")
    
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    
    to_process = []
    for rank, v in enumerate(videos, 1):
        vid, name, v_id, v_url, thumb_url, likes = v
        handle = clean_handle_name(name)
        out_name = f"{rank:03d}_{handle}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        clean_vid = extract_clean_video_id(v_url, v_id)
        
        needs_capture = True
        if os.path.exists(out_path):
            existing = cv2.imread(out_path)
            if existing is not None:
                h, w = existing.shape[:2]
                if min(h, w) >= 640:
                    score, _ = analyze_and_score_frame(detector, existing)
                    if score > 0:
                        needs_capture = False
                        
        if needs_capture:
            to_process.append((rank, vid, name, clean_vid, v_url, thumb_url, likes, out_path))

    print(f"Videos needing direct HD video screenshot: {len(to_process)} / {len(videos)}")
    
    if not to_process:
        print("All 500 videos already have verified direct HD video screenshots!")
        return

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        async def fetch_media(rank, vid, name, clean_vid, v_url, thumb_url, likes, out_path):
            page = await ctx.new_page()
            embed_url = f"https://www.instagram.com/reel/{clean_vid}/embed/" if "/reel/" in v_url else f"https://www.instagram.com/p/{clean_vid}/embed/"
            video_src = None
            img_src = None
            try:
                await page.goto(embed_url, wait_until='domcontentloaded', timeout=7000)
                await asyncio.sleep(0.4)
                video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                img_src = await page.evaluate("() => document.querySelector('.EmbeddedMediaImage')?.src || null")
            except Exception:
                pass
            finally:
                await page.close()
                
            # Process media
            saved = False
            if video_src:
                frame = get_best_frame_from_video(video_src)
                if frame is not None:
                    cv2.imwrite(out_path, frame)
                    print(f"[{rank:03d}/500] ✓ SAVED Video Screenshot {name}: shape {frame.shape}")
                    saved = True
                    
            if not saved and img_src:
                try:
                    resp = requests.get(img_src, timeout=5)
                    if resp.status_code == 200:
                        frame = get_best_frame_from_image(resp.content)
                        if frame is not None:
                            cv2.imwrite(out_path, frame)
                            print(f"[{rank:03d}/500] ✓ SAVED Embed Still {name}: shape {frame.shape}")
                            saved = True
                except Exception:
                    pass
                    
            if not saved and thumb_url:
                try:
                    resp = requests.get(thumb_url, timeout=5)
                    if resp.status_code == 200:
                        frame = get_best_frame_from_image(resp.content)
                        if frame is not None:
                            cv2.imwrite(out_path, frame)
                            print(f"[{rank:03d}/500] ✓ SAVED High-Res Still {name}: shape {frame.shape}")
                            saved = True
                except Exception:
                    pass

            if not saved:
                print(f"[{rank:03d}/500] ✗ Could not capture {name} ({clean_vid})")

        # Process in batches
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i+batch_size]
            tasks = [fetch_media(*item) for item in batch]
            await asyncio.gather(*tasks)
            print(f"Progress: {min(i+batch_size, len(to_process))}/{len(to_process)} completed...")
            
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(run_capture(batch_size=8))
