import asyncio
import os
import re
import cv2
import numpy as np
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

MORE_TAGS = ["speechcoach", "tedtalk", "publicspeaking", "confidencecoach", "careeradvice"]

def check_strict_talking(frame):
    h, w = frame.shape[:2]
    if h <= w: return False, 0
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: return False, 0
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.90: return False, 0
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    if area_pct < 4.0 or area_pct > 18.0: return False, 0
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 12: return False, 0
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.20: return False, 0
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y: return False, 0

    # Strict obstruction check: NO hands anywhere in the entire head & neck box!
    head_box = frame[max(0, fy - int(fh*0.2)):min(h, fy + int(fh*1.3)), max(0, fx - int(fw*0.2)):min(w, fx + int(fw*1.2))]
    if head_box.size == 0: return False, 0
    gray_box = cv2.cvtColor(head_box, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray_box, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel)) > 1500: return False, 0 # Any edge in face box = fail

    # Talking check: mouth must be open with oral cavity/teeth
    my = (rm_y + lm_y) // 2
    lower_gray = cv2.cvtColor(frame[nt_y:min(h, fy+fh), fx:min(w, fx+fw)], cv2.COLOR_BGR2GRAY)
    lip_roi = lower_gray[max(0, my - nt_y - 6):min(lower_gray.shape[0], my - nt_y + 10), :]
    if lip_roi.size == 0 or np.std(lip_roi) < 20: return False, 0

    # Sharpness
    sharpness = cv2.Laplacian(gray_box, cv2.CV_64F).var()
    if sharpness < 45: return False, 0

    score = conf * 100 + sharpness*0.1 + np.std(lip_roi) - nose_offset*50
    return True, score

async def find_more():
    found = []
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()

        for tag in MORE_TAGS:
            if len(found) >= 10: break
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            print(f"Scanning #{tag}...")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=6000)
                await asyncio.sleep(0.3)
                reels = await page.evaluate(r"""() => {
                    const anchors = document.querySelectorAll('a[href*="/reel/"]');
                    return Array.from(anchors).map(a => ({
                        href: a.href.split('?')[0],
                        author: (a.innerText || '').split('\n')[0].trim()
                    }));
                }""")

                for r in reels:
                    if len(found) >= 10: break
                    author = re.sub(r'[^a-zA-Z0-9._-]', '', r['author'].lstrip('@'))[:30]
                    if not author: continue
                    m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                    vid = m.group(1) if m else None
                    if not vid: continue

                    await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=5000)
                    await asyncio.sleep(0.3)
                    v_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                    if not v_src: continue

                    cap = cv2.VideoCapture(v_src)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    step = max(1, total // 40)
                    best_f = None
                    best_s = -1.0

                    for idx in range(0, min(total, 500), step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret or frame is None: continue
                        ok, s = check_strict_talking(frame)
                        if ok and s > best_s:
                            best_s = s
                            best_f = frame.copy()
                    cap.release()

                    if best_f is not None:
                        out_p = f"scratch/candidates_15/extra_{len(found)+1}_{author}.jpg"
                        cv2.imwrite(out_p, best_f)
                        found.append({
                            'author': author,
                            'vid': vid,
                            'url': r['href'],
                            'path': out_p
                        })
                        print(f"  ✓ Found Extra: @{author} -> {out_p} ({best_f.shape}, score: {best_s:.1f})")
            except Exception as e:
                print(f"Error on #{tag}: {e}")

        await ctx.close()

if __name__ == "__main__":
    asyncio.run(find_more())
