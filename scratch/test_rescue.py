import asyncio
import cv2
import os
import numpy as np
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.7)

async def test_creator():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()
        # DTy5H8UjDhV
        embed_url = 'https://www.instagram.com/reel/DTy5H8UjDhV/embed/'
        print("Navigating to:", embed_url)
        await page.goto(embed_url, wait_until='networkidle', timeout=15000)
        
        video_src = await page.evaluate('''() => {
            const v = document.querySelector('video');
            return v ? v.src : null;
        }''')
        print("Video SRC:", video_src[:80] if video_src else "None")
        await ctx.close()

        if video_src:
            cap = cv2.VideoCapture(video_src)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames in video:", total)
            step = max(1, total // 20)
            
            best_frame = None
            best_score = -1
            best_face = None

            for i in range(0, min(total, 400), step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                h, w = frame.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                if faces is None or len(faces) == 0:
                    continue
                face = max(faces, key=lambda f: f[2]*f[3])
                conf = float(face[-1])
                fw, fh = face[2], face[3]
                re_x, re_y = int(face[4]), int(face[5])
                le_x, le_y = int(face[6]), int(face[7])

                # Check eye openness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eye_w, eye_h = int(fw * 0.16), int(fh * 0.12)
                def get_eye_score(ex, ey):
                    x1, x2 = max(0, ex - eye_w//2), min(w, ex + eye_w//2)
                    y1, y2 = max(0, ey - eye_h//2), min(h, ey + eye_h//2)
                    patch = gray[y1:y2, x1:x2]
                    if patch.size == 0: return 0, 0
                    min_val, max_val, _, _ = cv2.minMaxLoc(patch)
                    sobel = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                    return (max_val - min_val), np.mean(np.abs(sobel))

                r_con, r_edge = get_eye_score(re_x, re_y)
                l_con, l_edge = get_eye_score(le_x, le_y)

                # Sharpness
                fx, fy = int(face[0]), int(face[1])
                face_crop = gray[max(0, fy):min(h, fy+int(fh)), max(0, fx):min(w, fx+int(fw))]
                sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0

                # Score: high conf, both eyes open (high contrast & edge), high sharpness
                if r_con > 50 and l_con > 50 and r_edge > 18 and l_edge > 18:
                    composite = conf * 100 + min(r_edge, l_edge) + min(r_con, l_con) * 0.2 + min(100, sharpness * 0.05)
                    if composite > best_score:
                        best_score = composite
                        best_frame = frame
                        best_face = face

            cap.release()

            if best_frame is not None:
                print("Found excellent unobstructed open-eye frame! Score:", best_score)
                fx, fy, fw, fh = [int(v) for v in best_face[0:4]]
                h, w = best_frame.shape[:2]
                pad_x = int(fw * 0.35)
                pad_y_top = int(fh * 0.50)
                pad_y_bot = int(fh * 0.35)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y_top)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y_bot)
                headshot = best_frame[y1:y2, x1:x2]
                cv2.imwrite("scratch/test_batch/creator____rescued.jpg", headshot)
                print("Saved rescued image to scratch/test_batch/creator____rescued.jpg")

asyncio.run(test_creator())
