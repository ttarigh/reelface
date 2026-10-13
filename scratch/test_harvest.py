import asyncio
import os
import re
import cv2
import numpy as np
import requests
import sqlite3
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

def clean_name(name):
    # Only the first line of author
    first_line = name.split('\n')[0].strip().lstrip('@')
    clean = re.sub(r'[^a-zA-Z0-9._-]', '', first_line)
    return clean[:30]

def validate_talking_head_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None: return -1.0, "No face"
    
    # Must have exactly 1 prominent face
    prominent = [f for f in faces if f[-1] > 0.70]
    if len(prominent) != 1:
        return -1.0, f"{len(prominent)} faces"
        
    face = prominent[0]
    conf = float(face[-1])
    if conf < 0.85: return -1.0, f"Low conf {conf:.2f}"
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    # Must be talking-head framing: bust / chest up (3.5% to 22%)
    if area_pct < 3.5 or area_pct > 22.0:
        return -1.0, f"Bad area {area_pct:.1f}%"
        
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 0: return -1.0, "Bad eye span"
    
    # Looking directly at camera (frontal eye contact)
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.28:
        return -1.0, f"Looking away (nose offset {nose_offset:.2f})"
        
    # Landmark order check: eyes above nose, nose above mouth
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, "Landmark order inversion"
        
    # Mouth separation
    mouth_w = abs(lm_x - rm_x)
    if mouth_w < eye_span * 0.40:
        return -1.0, "Mouth too narrow / occluded"

    # Check mouth and lower face for obstruction:
    # Lower face region from nose to chin:
    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, "Empty lower face"
    
    # In Simone Giertz, lower face was blocked by a dark t-shirt and straight cut edge
    # Check lower face color variance & edges:
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    # If a horizontal straight edge (prop/sign/hanger) cuts across, horizontal gradient has massive spike
    max_h_grad = np.max(np.abs(sobel_h))
    if max_h_grad > 1800: # strong artificial occlusion edge
        return -1.0, f"Occlusion edge {max_h_grad:.0f}"

    # Eye openness check
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
        return -1.0, f"Eyes closed/squinting (r_con={r_con}, l_con={l_con})"
        
    # Sharpness
    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 35:
        return -1.0, f"Blurry {sharpness:.1f}"

    score = conf * 100 + min(r_edge, l_edge)*1.5 + min(r_con, l_con)*0.3 + min(50, sharpness*0.05) - nose_offset*40
    return score, "OK"

os.makedirs("scratch/verified_talking_heads", exist_ok=True)

async def test_harvest():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        page = await ctx.new_page()
        
        tags = ["publicspeaking", "tedtalk", "communicationtips", "speechcoach", "dailywisdom"]
        verified = []
        
        for tag in tags:
            if len(verified) >= 10: break
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            print(f"\n--- Checking #{tag} ---")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=8000)
                await asyncio.sleep(0.5)
                reels = await page.evaluate(r"""() => {
                    const anchors = document.querySelectorAll('a[href*="/reel/"]');
                    return Array.from(anchors).map(a => {
                        const img = a.querySelector('img');
                        const text = a.innerText || '';
                        const parts = text.split('\n');
                        return {
                            href: a.href.split('?')[0],
                            author: (parts[0] || '').trim(),
                            views: (parts[1] || '').trim(),
                            caption: parts.slice(2).join(' ').trim()
                        };
                    });
                }""")
                
                for r in reels:
                    if len(verified) >= 10: break
                    author = clean_name(r['author'])
                    if not author or any(b in author.lower() for b in ['movie', 'film', 'clip', 'quotes', 'fact', 'ai', 'page', 'hub', 'tv', 'news', 'baker', 'art']):
                        continue
                    m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                    vid = m.group(1) if m else None
                    if not vid: continue
                    
                    # Check embed
                    embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
                    try:
                        await page.goto(embed_url, wait_until='domcontentloaded', timeout=6000)
                        await asyncio.sleep(0.35)
                        video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                        if not video_src: continue
                        
                        # Sample frames
                        cap = cv2.VideoCapture(video_src)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        step = max(1, total // 25)
                        best_frame = None
                        best_score = -1.0
                        
                        for idx in range(0, min(total, 450), step):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if not ret or frame is None: continue
                            s, msg = validate_talking_head_frame(frame)
                            if s > best_score:
                                best_score = s
                                best_frame = frame.copy()
                        cap.release()
                        
                        if best_frame is not None:
                            out_p = f"scratch/verified_talking_heads/{len(verified)+1:03d}_{author}.jpg"
                            cv2.imwrite(out_p, best_frame)
                            print(f"  ✓ VERIFIED TALKING HEAD: @{author} -> {out_p} ({best_frame.shape}, score: {best_score:.1f})")
                            verified.append((author, vid, r['caption'][:60]))
                        else:
                            print(f"  ✗ Rejected @{author}: failed strict talking-head validation")
                    except Exception as e:
                        pass
            except Exception as e:
                print(f"Error on #{tag}: {e}")
                
        await ctx.close()
        print(f"\nSuccessfully harvested and verified {len(verified)} pure talking-head creators!")

asyncio.run(test_harvest())
