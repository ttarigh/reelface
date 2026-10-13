"""
Real-time Streaming Direct Video Screenshot Generator
- 16 parallel workers.
- Each worker loads embed, extracts MP4 video.src, samples frames,
  validates solitary talking head (eyes open, unobstructed face, frontal gaze),
  and writes talking_head_photos/{rank:03d}_{handle}.jpg immediately.
- If video stream fails, falls back to direct unwatermarked high-res thumbnail.
"""

import asyncio
import os
import re
import sqlite3
import threading
import cv2
import numpy as np
import requests
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

thread_local = threading.local()

def get_detector():
    if not hasattr(thread_local, "detector"):
        thread_local.detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    return thread_local.detector

def clean_handle(handle):
    h = handle.split('\n')[0].strip().lstrip('@')
    clean = re.sub(r'[^a-zA-Z0-9._-]', '', h)
    return clean[:30]

def validate_frame(frame):
    h, w = frame.shape[:2]
    detector = get_detector()
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None: return -1.0, None
    
    prominent = [f for f in faces if f[-1] > 0.65]
    if len(prominent) != 1: return -1.0, None
    
    face = prominent[0]
    conf = float(face[-1])
    if conf < 0.80: return -1.0, None
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    if area_pct < 2.5 or area_pct > 26.0: return -1.0, None
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0: return -1.0, None
    
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.32: return -1.0, None
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y: return -1.0, None
    if abs(lm_x - rm_x) < eye_span * 0.35: return -1.0, None

    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, None
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel_h)) > 2000: return -1.0, None

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
    if r_con < 32 or l_con < 32 or r_edge < 10 or l_edge < 10: return -1.0, None
        
    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 28: return -1.0, None

    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40
    return score, face

def sample_video(video_src):
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 20) if total > 0 else 15
    best_f = None
    best_s = -1.0
    for idx in range(0, min(total, 400), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        s, _ = validate_frame(frame)
        if s > best_s:
            best_s = s
            best_f = frame.copy()
    cap.release()
    return best_f

async def run_streaming_capture():
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT id, creator_name, video_id, video_url, like_count, thumbnail_url FROM videos ORDER BY like_count DESC LIMIT 500")
    videos = c.fetchall()
    conn.close()

    print(f"Loaded {len(videos)} videos from database.")
    saved_count = 0
    lock = asyncio.Lock()

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )

        sem = asyncio.Semaphore(12)

        async def process_video(rank, row):
            nonlocal saved_count
            vid_id, creator_name, video_id, video_url, likes, thumb_url = row
            handle = clean_handle(creator_name)
            out_p = os.path.join(OUTPUT_DIR, f"{rank:03d}_{handle}.jpg")

            if os.path.exists(out_p) and os.path.getsize(out_p) > 20000:
                async with lock:
                    saved_count += 1
                return

            v_src = None
            async with sem:
                page = await ctx.new_page()
                try:
                    await page.goto(f"https://www.instagram.com/reel/{video_id}/embed/", wait_until='domcontentloaded', timeout=6000)
                    await asyncio.sleep(0.3)
                    v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                except Exception:
                    pass
                finally:
                    await page.close()

            best_f = None
            if v_src:
                best_f = await asyncio.to_thread(sample_video, v_src)

            if best_f is not None:
                cv2.imwrite(out_p, best_f)
                async with lock:
                    saved_count += 1
                    if saved_count % 20 == 0 or rank <= 20:
                        print(f"  [{saved_count}/500] Rank {rank:03d}: @{handle} -> DIRECT VIDEO SCREENSHOT {best_f.shape}")
            else:
                # Fallback to direct HD thumbnail download
                if thumb_url and thumb_url.startswith('http'):
                    def dl_thumb():
                        try:
                            resp = requests.get(thumb_url, timeout=5)
                            if resp.status_code == 200:
                                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    cv2.imwrite(out_p, img)
                                    return True
                        except Exception:
                            pass
                        return False
                    ok = await asyncio.to_thread(dl_thumb)
                    if ok:
                        async with lock:
                            saved_count += 1
                            if saved_count % 20 == 0:
                                print(f"  [{saved_count}/500] Rank {rank:03d}: @{handle} -> HQ THUMB")

        # Process in chunks of 50
        chunk_size = 50
        for i in range(0, len(videos), chunk_size):
            chunk = videos[i:i+chunk_size]
            tasks = [process_video(i + idx + 1, v) for idx, v in enumerate(chunk)]
            await asyncio.gather(*tasks)
            print(f"Processed chunk {i//chunk_size + 1}/{len(videos)//chunk_size + 1}. Total saved: {saved_count}/500")

        await ctx.close()

    print(f"\nFinished! Total photos saved in {OUTPUT_DIR}: {len(os.listdir(OUTPUT_DIR))}/500")

if __name__ == "__main__":
    asyncio.run(run_streaming_capture())
