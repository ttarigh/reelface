import asyncio
import cv2
import numpy as np
import os
import requests
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

test_cases = [
    ("kylenutt", "C0t4ot2uPWd"),
    ("creator___", "DTy5H8UjDhV"),
    ("with.kuttippa", "Dci6WMxOxOQ"),
    ("yurii.thebuilder", "DbtTfhphjMu"),
    ("fazerug", "DdSpPw5tbWf")
]

os.makedirs("scratch/test_direct_screenshots", exist_ok=True)

async def test_direct_screenshots():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        for name, vid in test_cases:
            page = await ctx.new_page()
            url = f"https://www.instagram.com/reel/{vid}/embed/"
            print(f"Opening {name} ({url})...")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=8000)
                await asyncio.sleep(0.8)
                video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                img_src = await page.evaluate("() => document.querySelector('.EmbeddedMediaImage')?.src || null")
            except Exception as e:
                print(f"Error loading {name}: {e}")
                video_src, img_src = None, None
            finally:
                await page.close()

            print(f"[{name}] video_src: {bool(video_src)}, img_src: {bool(img_src)}")
            
            best_frame = None
            best_score = -1

            if video_src:
                cap = cv2.VideoCapture(video_src)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 20)
                print(f"[{name}] total video frames: {total}")
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
                    
                    # Check eyes and symmetry
                    re_x, re_y = int(face[4]), int(face[5])
                    le_x, le_y = int(face[6]), int(face[7])
                    nt_x, nt_y = int(face[8]), int(face[9])
                    eye_span = abs(le_x - re_x)
                    if eye_span <= 0: continue
                    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
                    if nose_offset > 0.35: continue # must be looking directly at camera
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    eye_w, eye_h = int(face[2]*0.16), int(face[3]*0.12)
                    def eye_contrast(ex, ey):
                        x1, x2 = max(0, ex - eye_w//2), min(w, ex + eye_w//2)
                        y1, y2 = max(0, ey - eye_h//2), min(h, ey + eye_h//2)
                        patch = gray[y1:y2, x1:x2]
                        if patch.size == 0: return 0
                        min_v, max_v, _, _ = cv2.minMaxLoc(patch)
                        return max_v - min_v
                    
                    r_con = eye_contrast(re_x, re_y)
                    l_con = eye_contrast(le_x, le_y)
                    if r_con < 45 or l_con < 45: continue # eyes must be open
                    
                    # Sharpness
                    fx, fy, fw, fh = [int(v) for v in face[:4]]
                    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
                    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
                    
                    score = conf * 100 + min(r_con, l_con) * 0.5 + min(100, sharpness * 0.1) - nose_offset * 50
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
                cap.release()

            if best_frame is not None:
                out_path = f"scratch/test_direct_screenshots/{name}_video_screenshot.jpg"
                cv2.imwrite(out_path, best_frame)
                print(f"SAVED {out_path}, shape: {best_frame.shape}, score: {best_score:.1f}")
            else:
                print(f"Could not find valid video frame for {name}")

        await ctx.close()

asyncio.run(test_direct_screenshots())
