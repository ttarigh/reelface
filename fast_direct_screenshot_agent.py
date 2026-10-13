"""
Fast Direct Video Screenshot Agent
- Runs 16 concurrent browser tabs to resolve media URLs.
- Uses ThreadPoolExecutor to sample direct video frames in parallel.
- Selects optimal frame with eyes open, frontal gaze, unobstructed face.
- Saves pristine HD direct screenshots without play buttons.
"""

import asyncio
import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import time
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

def clean_handle_name(name):
    clean = re.sub(r'[^a-zA-Z0-9._-]', '', name.lstrip('@'))
    return clean[:30]

def extract_clean_video_id(v_url, v_id):
    m = re.search(r'/(?:reel|p)/([a-zA-Z0-9_-]+)', v_url)
    if m: return m.group(1)
    m2 = re.search(r'([a-zA-Z0-9_-]{9,})', v_id)
    if m2: return m2.group(1)
    return v_id

def analyze_and_score_frame(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return -1.0, None
        
    face = max(faces, key=lambda f: f[2]*f[3])
    conf = float(face[-1])
    if conf < 0.78:
        return -1.0, None
        
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    if fw < 40 or fh < 40:
        return -1.0, None

    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0:
        return -1.0, None
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.45:
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
    
    # Eyes open check
    if r_con < 25 or l_con < 25:
        return -1.0, None

    # Sharpness
    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 18:
        return -1.0, None

    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*30
    return score, face

def process_video_stream(video_src, max_samples=25):
    detector = get_detector()
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

def process_image_url(img_url):
    detector = get_detector()
    try:
        resp = requests.get(img_url, timeout=5)
        if resp.status_code != 200: return None
        nparr = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return None
        if min(frame.shape[:2]) < 600: return None
        
        score, _ = analyze_and_score_frame(detector, frame)
        if score > 0:
            return frame
    except Exception:
        pass
    return None

async def resolve_all_embed_urls(videos_to_fetch, concurrency=16):
    results = {}
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        sem = asyncio.Semaphore(concurrency)
        
        async def fetch_one(vid, name, clean_vid, v_url):
            async with sem:
                page = await ctx.new_page()
                embed_url = f"https://www.instagram.com/reel/{clean_vid}/embed/" if "/reel/" in v_url else f"https://www.instagram.com/p/{clean_vid}/embed/"
                v_src, i_src = None, None
                is_dead = False
                try:
                    await page.goto(embed_url, wait_until='domcontentloaded', timeout=6500)
                    await asyncio.sleep(0.35)
                    body_text = await page.evaluate("() => document.body.innerText || ''")
                    if "may be broken, or the post may have been removed" in body_text:
                        is_dead = True
                    else:
                        v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                        i_src = await page.evaluate("() => document.querySelector('.EmbeddedMediaImage')?.src || null")
                except Exception:
                    pass
                finally:
                    await page.close()
                results[clean_vid] = (v_src, i_src, is_dead)

        tasks = [fetch_one(v[0], v[1], v[2], v[3]) for v in videos_to_fetch]
        await asyncio.gather(*tasks)
        await ctx.close()
    return results

def run():
    conn = sqlite3.connect("viral_videos.db")
    c = conn.cursor()
    c.execute("SELECT id, creator_name, video_id, video_url, thumbnail_url, like_count FROM videos ORDER BY like_count DESC")
    all_videos = c.fetchall()
    
    # Load reserve candidates if needed
    reserves = []
    if os.path.exists("scratch/reserve_candidates.json"):
        with open("scratch/reserve_candidates.json") as f:
            reserves = json.load(f)

    # Check which ones need capture
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    
    to_process = []
    for rank, v in enumerate(all_videos, 1):
        vid, name, v_id, v_url, thumb_url, likes = v
        handle = clean_handle_name(name)
        out_name = f"{rank:03d}_{handle}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        clean_vid = extract_clean_video_id(v_url, v_id)
        
        needs_capture = True
        if os.path.exists(out_path):
            existing = cv2.imread(out_path)
            if existing is not None and min(existing.shape[:2]) >= 640:
                score, _ = analyze_and_score_frame(detector, existing)
                if score > 0:
                    needs_capture = False
                    
        if needs_capture:
            to_process.append((vid, name, clean_vid, v_url, thumb_url, rank, out_path))

    print(f"Total videos: {len(all_videos)}, Needing direct HD screenshot: {len(to_process)}")
    if not to_process:
        print("All photos already captured!")
        return

    # Process in batches of 40 to avoid memory buildup
    batch_size = 32
    for b_idx in range(0, len(to_process), batch_size):
        batch = to_process[b_idx:b_idx+batch_size]
        print(f"\n--- Batch {b_idx//batch_size + 1} ({len(batch)} items) ---")
        
        # Step 1: Resolve embed URLs with Playwright
        embed_data = asyncio.run(resolve_all_embed_urls(batch, concurrency=16))
        
        # Step 2: Extract direct video screenshots in parallel
        def handle_item(item):
            vid, name, clean_vid, v_url, thumb_url, rank, out_path = item
            v_src, i_src, is_dead = embed_data.get(clean_vid, (None, None, False))
            
            frame = None
            if v_src:
                frame = process_video_stream(v_src)
                if frame is not None:
                    cv2.imwrite(out_path, frame)
                    print(f"[{rank:03d}/500] ✓ SAVED Direct Video Screenshot {name}: shape {frame.shape}")
                    return True
                    
            if frame is None and i_src:
                frame = process_image_url(i_src)
                if frame is not None:
                    cv2.imwrite(out_path, frame)
                    print(f"[{rank:03d}/500] ✓ SAVED Direct Embed Still {name}: shape {frame.shape}")
                    return True
                    
            # Fallback if dead or not found: substitute with reserve
            if is_dead or frame is None:
                if reserves:
                    rep = reserves.pop(0)
                    r_handle = clean_handle_name(rep['creator_name'])
                    print(f"[{rank:03d}/500] 🔄 Replacing inactive {name} with verified reserve {rep['creator_name']}")
                    # Update DB
                    c.execute("""UPDATE videos SET creator_name=?, creator_url=?, video_id=?, video_url=?, title=?, view_count=?, like_count=?, thumbnail_url=? WHERE id=?""",
                              (rep['creator_name'], rep['creator_url'], rep['video_id'], rep['video_url'], rep['title'], rep['view_count'], rep['like_count'], rep['thumbnail_url'], vid))
                    conn.commit()
                    # Capture replacement directly
                    new_out = os.path.join(OUTPUT_DIR, f"{rank:03d}_{r_handle}.jpg")
                    # Try reserve thumbnail if high res
                    if rep.get('thumbnail_url'):
                        rf = process_image_url(rep['thumbnail_url'])
                        if rf is not None:
                            cv2.imwrite(new_out, rf)
                            print(f"[{rank:03d}/500] ✓ SAVED Reserve Still {rep['creator_name']}: shape {rf.shape}")
                            return True
            return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(handle_item, batch))

    conn.close()
    print("\nCapture run complete!")

if __name__ == "__main__":
    run()
